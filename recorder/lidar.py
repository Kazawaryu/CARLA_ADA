#!/usr/bin/python3

import re
import sys
import time
import open3d as o3d
import cv2
import math
import carla
import numpy as np

from recorder.sensor import Sensor


class Lidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        # Save as a Nx4 numpy array. Each row is a point (x, y, z, intensity)
        lidar_data = np.fromstring(bytes(sensor_data.raw_data),
                                   dtype=np.float32)
        lidar_data = np.reshape(
            lidar_data, (int(lidar_data.shape[0] / 4), 4))

        # Convert point cloud to right-hand coordinate system
        # lidar_data[:, 1] *= -1

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].npy
        # np.save("{}/{:0>10d}".format(save_dir, sensor_data.frame), lidar_data)
        with open("{}/{:0>10d}.bin".format(save_dir,sensor_data.frame), 'wb') as file:
            file.write(lidar_data)
            
        return True


class SemanticLidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        self.load_parameters()

    def load_parameters(self):      
        # TODO: write parameters in config file(.yaml)
        self.Label_dict ={14:"Car",15:"Truck",16:"Bus"}
        self.Active = True
        self.Largest_label_range = 75       # object labeling range, max 100
            

        # ===== newest version (9.16) parameters =====

        # L1 parameters
        self.PC_MAX_RANGE = 60
        self.PC_NUM_RING = 60
        self.PC_NUM_SECTOR = 60
        self.PC_ENTROPY_SCORE_LIMIT = 0.4


        if self.Active:
            self._Hs = 0.8                  # scene entropy 

            self._rho_b = 25                # temp object rho 
            self._rho_s = 5                # temp object rho 
            self._f_tra = 1.2                 # tracking task 

            self._k_sig = 4                 # parameter value sigimoid 
            self._f_sig = 0.8               # result value sigimoid

            self.last_entropy = 3
 
            # self.load_object_detection_model()

            self.last_trans = {}
            


    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        # Save data as a Nx6 numpy array.
        lidar_data = np.fromstring(bytes(sensor_data.raw_data),
                                   dtype=np.dtype([
                                       ('x', np.float32),
                                       ('y', np.float32),
                                       ('z', np.float32),
                                       ('CosAngle', np.float32),
                                       ('ObjIdx', np.uint32),
                                       ('ObjTag', np.uint32)
                                   ]))

        # Convert point cloud to right-hand coordinate system
        # lidar_data['y'] *= -1

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].npy

        if self.Active:
            result, labels, score = self.one_loop_cal_all_active_old(lidar_data)
            if result:
                self.save_data(save_dir,sensor_data,lidar_data,labels)
        else:
            labels,_ = self.get_label(lidar_data)
            self.save_data(save_dir,sensor_data,lidar_data,labels)
        return True
    

    def save_data(self,save_dir,sensor_data,lidar_data,labels):
        with open("{}/{:0>10d}.bin".format(save_dir,sensor_data.frame), 'wb') as file:
                file.write(lidar_data)
        with open("{}/{:0>10d}.txt".format(save_dir,sensor_data.frame),'a+') as f:
            for line in labels:
                print(line,file=f)

    def get_label(self,lidar_data):
        labels = []
        objects_dict,ground = self.get_label_centerpoint(lidar_data)
        bbox_dict,trans_dict,tags_dict,sensor_trans = self.get_near_boudning_box_by_world()
        for key in bbox_dict:
            if key in objects_dict.keys():
                temp_bbox = bbox_dict[key]
                temp_points = objects_dict[key]
                temp_points = np.array([list(elem) for elem in temp_points])

                max_p = np.max(temp_points, axis=0)
                min_p = np.min(temp_points, axis=0)
                temp_bbox = bbox_dict[key]
                temp_trans = trans_dict[key]
                temp_tag = tags_dict[key]

                np.mean(max_p[0] + min_p[0])
                cx = (max_p[0] + min_p[0])/2
                cy = (max_p[1] + min_p[1])/2
                cz = (temp_trans.location.z - sensor_trans.location.z +temp_bbox.location.z)

                sx = 2*temp_bbox.extent.x
                sy = 2*temp_bbox.extent.y
                sz = 2*temp_bbox.extent.z
                yaw = (temp_trans.rotation.yaw - sensor_trans.rotation.yaw + temp_bbox.rotation.yaw)

                label_str = "{} {} {} {} {} {} {} {}" .format(cx, cy, cz, sx, sy, sz, yaw, self.Label_dict[temp_tag[0]])
    
                labels.append(label_str)
        
        return labels,ground
    
    def get_label_centerpoint(self,semantic_points):
        objects_dict = {}
        ground = []
        for point in semantic_points:
            if point[5] in self.Label_dict.keys():
                if not point[4] in objects_dict:
                    objects_dict[point[4]] = []
                objects_dict[point[4]].append(point)
            elif point[5] == 1:
                ground.append(point)


        return objects_dict,ground

    def set_world(self, world):
        self.world = world
        

    def get_near_boudning_box_by_world(self):
        bbox_dict = {}
        trans_dict={}
        tags_dict = {}

        actors_list = self.world.get_actors()
        for actor in actors_list:
            if re.match("^vehicle",str(actor.type_id)):
                dist = actor.get_transform().location.distance(self.carla_actor.get_transform().location)
                if dist < self.Largest_label_range:
                    bbox_dict[actor.id] = actor.bounding_box
                    trans_dict[actor.id] = actor.get_transform()
                    tags_dict[actor.id] = actor.semantic_tags
        
        return bbox_dict,trans_dict,tags_dict,self.carla_actor.get_transform()
    
    # ============== Active Startegy (OLD) ===================

    
    def cal_entropy_if_keep(self, entropy_last, entropy_now):
        return abs(entropy_now - entropy_last) / -entropy_last >= self._Hs

    def cal_sigmoid(self, max_dist, det_dist):
        x = det_dist/ max_dist
        y = (1 - np.power(np.e, -self._k_sig * x) )/(1 + np.power(np.e, -self._k_sig * x))

        return 1 - y 


    def cal_3d_iou(self,corners1, corners2):
        iou = 0.76
        iou_2d = 0.8

        return iou,iou_2d


    def one_loop_cal_all_active_old(self,lidar_data):
        # 1. Initialize lidar_data
        tick = time.time()
        semantic_points = np.array([list(elem) for elem in lidar_data])

        # 2. Calculate the scene entropy
        voxel_size = 2
        voxel_max_range = np.max(semantic_points[:,:3], axis=0)
        voxel_min_range = np.min(semantic_points[:,:3], axis=0)

        
        # 3. Get label center point
        objects_dict = {}
        entropy = 0

        voxel_count = np.ceil((voxel_max_range - voxel_min_range) / voxel_size).astype(int)[:3]
        voxel_scene = np.zeros(voxel_count)
        
        indices = np.floor((semantic_points[:,:3] - voxel_min_range) / voxel_size).astype(int)

        # TODO: speed up code here

        for i in indices:
            voxel_scene[i[0],i[1],i[2]] += 1
        # np.add.at(voxel_scene, (indices[:,0], indices[:,1], indices[:,2]), 1)

        
        
        dt = len(semantic_points) / ((voxel_max_range[0] - voxel_min_range[0]) * (voxel_max_range[1] - voxel_min_range[1]) * (voxel_max_range[2] - voxel_min_range[2]))

        nonzero_indices = np.nonzero(voxel_scene)
        di = voxel_scene[nonzero_indices]
        entropy = -np.sum((di / dt) * np.log10(di / dt))

        valid_labels = np.isin(semantic_points[:, 5], list(self.Label_dict.keys()))
        valid_points = semantic_points[valid_labels]
        unique_labels = np.unique(valid_points[:, 4])

        for label in unique_labels:
            objects_dict[int(label)] = valid_points[valid_points[:, 4] == label]
        
        ground_indices = np.where(semantic_points[:, 5] == 1)
        ground = semantic_points[ground_indices]
        
        if self.last_entropy == 3:
            self.last_entropy = entropy
        elif self.cal_entropy_if_keep(self.last_entropy, entropy) or entropy < -20000:
            self.last_entropy = entropy
        else:
            return False, None, None

            
        # 4. Calculate the area point rho
        
        labels = []
        now_trans_dict = {}

        actors_list = self.world.get_actors()
        for actor in actors_list:
            if actor.id in objects_dict.keys() and re.match("^vehicle",str(actor.type_id)):
                dist = actor.get_transform().location.distance(self.carla_actor.get_transform().location)

                if dist < self.Largest_label_range:
                    bbox = actor.bounding_box
                    points_collection = objects_dict[actor.id]
                    count = len(points_collection)
                    volume = 8 * bbox.extent.x * bbox.extent.y * bbox.extent.z
                    rho = count / volume

                    if rho > self._rho_b:

                        points_collection = np.array([list(elem) for elem in points_collection])
                        max_p = np.max(points_collection, axis=0)
                        min_p = np.min(points_collection, axis=0)
                        cx = (max_p[0] + min_p[0])/2
                        cy = (max_p[1] + min_p[1])/2
                        cz = (actor.get_transform().location.z - self.carla_actor.get_transform().location.z + bbox.location.z)

                        sx = 2*bbox.extent.x
                        sy = 2*bbox.extent.y
                        sz = 2*bbox.extent.z
                        yaw = (actor.get_transform().rotation.yaw - self.carla_actor.get_transform().rotation.yaw + bbox.rotation.yaw)

                        label_str = "{} {} {} {} {} {} {} {}" .format(cx, cy, cz, sx, sy, sz, yaw, self.Label_dict[actor.semantic_tags[0]])
                        labels.append(label_str)
                        now_trans_dict[actor.id] = [cx,cy]
                        
                    elif rho > self._rho_s:
                        points_collection = np.array([list(elem) for elem in points_collection])
                        max_p = np.max(points_collection, axis=0)
                        min_p = np.min(points_collection, axis=0)
                        cx = (max_p[0] + min_p[0])/2
                        cy = (max_p[1] + min_p[1])/2

                        now_trans_dict[actor.id] = [cx,cy]
                        if actor.id in self.last_trans.keys():
                      
                            delta_trans = [self.last_trans[actor.id][0]-cx, self.last_trans[actor.id][1]-cy]
                            delta_dist = np.sqrt(delta_trans[0]**2 + delta_trans[1]**2)
                        
                            if delta_dist > self._f_tra:
                                cz = (actor.get_transform().location.z - self.carla_actor.get_transform().location.z + bbox.location.z)

                                sx = 2*bbox.extent.x
                                sy = 2*bbox.extent.y
                                sz = 2*bbox.extent.z
                                yaw = (actor.get_transform().rotation.yaw - self.carla_actor.get_transform().rotation.yaw + bbox.rotation.yaw)

                                label_str = "{} {} {} {} {} {} {} {}" .format(cx, cy, cz, sx, sy, sz, yaw, self.Label_dict[actor.semantic_tags[0]])
                                labels.append(label_str)
                                
                            else:
                                print("[delta_dist]", delta_dist)
        
        self.last_trans = now_trans_dict
        # print("###[LABEL FINISH]###", time.time() - tick,"###[entropy]###", entropy)
        
        return True, labels, 100

    # ============== Active Startegy (916) ===================
    
    def scene_entropy(self,desc,pcd):
        max_pcd = np.max(pcd,axis=0)
        min_pcd = np.min(pcd,axis=0)
        vt = len(pcd) / (max_pcd[0]-min_pcd[0])*(max_pcd[1]-min_pcd[1])*(max_pcd[2]-min_pcd[2])
        nonzero_indices = np.nonzero(desc)
        vi = desc[nonzero_indices]
        entropy = -np.sum((vi /vt) * np.log(vi /vt))

        return entropy

        
    def one_loop_cal_all_active_916(self,lidar_data):
        # 0. Initialize lidar_data
        semantic_point = np.array([list(elem) for elem in lidar_data])

        # ========= L1 - Traffic scene complexity ==========
        # 1.1 scan and bev context build
        scan_desc = np.zeros((self.PC_NUM_RING, self.PC_NUM_SECTOR))
        bev_max = np.zeros((self.PC_MAX_RANGE, self.PC_MAX_RANGE))
        bev_min = np.zeros((self.PC_MAX_RANGE, self.PC_MAX_RANGE))
        pt_range = self.PC_MAX_RANGE / 2

        pt_x = semantic_point[:, 0]
        pt_y = semantic_point[:, 1]
        pt_z = semantic_point[:, 2]

        azim_range = np.sqrt(pt_x ** 2 + pt_y ** 2)
        azim_angle = np.rad2deg(np.arctan2(pt_y, pt_x))
        azim_angle[azim_angle < 0] += 360

        valid_indices = np.where(azim_range < self.PC_MAX_RANGE) 
        azim_sector = np.floor(azim_angle[valid_indices] / (360 / self.PC_NUM_SECTOR)).astype(np.int32)
        azim_ring = np.floor(azim_range[valid_indices] / (self.PC_MAX_RANGE / self.PC_NUM_RING)).astype(np.int32)

        np.add.at(scan_desc, (azim_ring, azim_sector), 1)

        valid_indices = np.where((pt_x < pt_range) & (pt_x > -pt_range) & (pt_y < pt_range) & (pt_y > -pt_range))
        pt_x_valid = pt_x[valid_indices] + pt_range
        pt_y_valid = pt_y[valid_indices] + pt_range
        pt_z_valid = pt_z[valid_indices]

        bev_max_indices = (pt_x_valid.astype(int), pt_y_valid.astype(int))
        np.maximum.at(bev_max, bev_max_indices, pt_z_valid)

        bev_min_indices = (pt_x_valid.astype(int), pt_y_valid.astype(int))
        np.minimum.at(bev_min, bev_min_indices, pt_z_valid)

        bev_scan = np.subtract(bev_max, bev_min)

        # 1.2  two kind context entropy score calculate

        scan_entropy = self.scene_entropy(scan_desc, semantic_point[:, :3])
        bev_entropy = self.scene_entropy(bev_scan, semantic_point[:, :3])
        current_entropy_score = bev_entropy / scan_entropy


        # ========= L2 - Traffic scene complexity ==========
        # 2.1 get moving object and center point (in fact, we use the ground truth from carla)  
        actor_points, current_trans, label_output = {}, {}, []

        valid_labels = np.isin(semantic_point[:, 5], list(self.Label_dict.keys()))
        valid_points = semantic_point[valid_labels]
        unique_labels = np.unique(valid_points[:, 4])
        for label in unique_labels:
            actor_points[int(label)] = valid_points[valid_points[:, 4] == label]
        
        actor_list = self.world.get_actors()
        filtered_actors = [actor for actor in actor_list if actor.id in actor_points.keys() and re.match("^vehicle", str(actor.type_id))]
        actor_dist = [actor.get_transform().location.distance(self.carla_actor.get_transform().location) for actor in actor_list]

        selected_actors = [actor for actor, dist in zip(filtered_actors, actor_dist) if dist < self.Largest_label_range]

        # 2.2 make a map to struct the actors, and give a basic frequency








        # ========= L3 - Uncertainty complexity ==========
        # 3.1 calculate the uncertainty (temp tracking) of each object

        carla_actor_transform = self.carla_actor.get_transform().location
        carla_actor_rotation_yaw = self.carla_actor.get_transform().rotation.yaw
        for actor in selected_actors:
            bbox = actor.bounding_box
            actor_id = actor.id
            points_collection = actor_points[actor_id]
            count = len(points_collection)
            volume = 8 * bbox.extent.x * bbox.extent.y * bbox.extent.z
            rho = count / volume

            if rho > self._rho_b: # TODO: times a attenuation parameter here
                points_collection = np.array([list(elem) for elem in points_collection])
                max_p = np.max(points_collection, axis=0)
                min_p = np.min(points_collection, axis=0)
                cx = (max_p[0] + min_p[0]) / 2
                cy = (max_p[1] + min_p[1]) / 2
                cz = (actor.get_transform().location.z - carla_actor_transform.z + bbox.location.z)
                sx = 2 * bbox.extent.x
                sy = 2 * bbox.extent.y
                sz = 2 * bbox.extent.z
                yaw = (actor.get_transform().rotation.yaw - carla_actor_rotation_yaw + bbox.rotation.yaw)

                label_str = "{} {} {} {} {} {} {} {}".format(cx, cy, cz, sx, sy, sz, yaw, self.Label_dict[actor.semantic_tags[0]])
                label_output.append(label_str)
                current_trans[actor_id] = [cx, cy]

            elif rho > self._rho_s: # TODO: times a attenuation parameter here
                points_collection = np.array([list(elem) for elem in points_collection])
                max_p = np.max(points_collection, axis=0)
                min_p = np.min(points_collection, axis=0)
                cx = (max_p[0] + min_p[0]) / 2
                cy = (max_p[1] + min_p[1]) / 2

                current_trans[actor_id] = [cx, cy]
                if actor.id in self.last_trans.keys():
                    delta_trans = [self.last_trans[actor.id][0]-cx, self.last_trans[actor.id][1]-cy]
                    delta_dist = np.sqrt(delta_trans[0]**2 + delta_trans[1]**2)

                    if delta_dist > self._f_tra:
                        cz = (actor.get_transform().location.z - carla_actor_transform.z + bbox.location.z)
                        sx = 2*bbox.extent.x
                        sy = 2*bbox.extent.y
                        sz = 2*bbox.extent.z
                        yaw = (actor.get_transform().rotation.yaw - self.carla_actor.get_transform().rotation.yaw + bbox.rotation.yaw)
                        
                        label_str = "{} {} {} {} {} {} {} {}".format(cx, cy, cz, sx, sy, sz, yaw, self.Label_dict[actor.semantic_tags[0]])
                        label_output.append(label_str)
        
        # 3.2 update last lidar frame data
        self.last_trans = current_trans
        self.last_entropy = {scan_entropy, bev_entropy, current_entropy_score}

        # ========= L4 - Algorithm feature complexity ==========






