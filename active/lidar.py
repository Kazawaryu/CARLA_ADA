import re
import sys
import time
import open3d as o3d
import cv2
import math
import carla
import numpy as np

class ActiveLidar:
    def __init__(self,world,sensor) -> None:
        self.world = world
        self.carla_actor = sensor
        self.set_parameters()
        

    def set_parameters(self):
        self.Walker_tags = {12:"Pedestrian",13:"Rider"}
        self.Vehicle_tags ={14:"Car",15:"Truck",16:"Bus"}
        self.Cyclist_tags = {18:"Motorcycle",19:"Bicycle"}
        self.Active = True
        self.Largest_label_range = 100       # object labeling range, max 100
            

        # ===== newest version (9.16) parameters =====

        # L1 parameters
        self.PC_MAX_RANGE = 60
        self.PC_NUM_RING = 60
        self.PC_NUM_SECTOR = 60
        self.PC_ENTROPY_SCORE_LIMIT = 0.4

        # L2 parameters
        self.LD_STD_NOISE = 0.2


        if self.Active:
            self._Hs = 0.8                  # scene entropy 

            self._rho_b = 25                # temp object rho 
            self._rho_s = 0                # temp object rho 
            self._f_tra = 1.2                 # tracking task 

            self._k_sig = 4                 # parameter value sigimoid 
            self._f_sig = 0.8               # result value sigimoid

            self.last_entropy = 3
 
            # self.load_object_detection_model()

            self.last_trans = {}

    def one_loop_cal_all_active_old(self,lidar_data):
        # 1. Initialize lidar_data
        tick = time.time()
        semantic_points = np.array([list(elem) for elem in lidar_data])

        labels = []
        now_trans_dict = {}
        objects_dict = {}

        valid_labels = np.isin(semantic_points[:, 5], list(self.Vehicle_tags.keys()))
        valid_points = semantic_points[valid_labels]
        unique_labels = np.unique(valid_points[:, 4])

        for label in unique_labels:
            objects_dict[int(label)] = valid_points[valid_points[:, 4] == label]

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

                    label_str = "{} {} {} {} {} {} {} {}" .format(cx, cy, cz, sx, sy, sz, yaw, self.Vehicle_tags[actor.semantic_tags[0]])
                    labels.append(label_str)
                    now_trans_dict[actor.id] = [cx,cy]

        self.last_trans = now_trans_dict

        return True, labels, 100
        
    def scene_entropy(self,desc,pcd):
        max_pcd = np.max(pcd,axis=0)
        min_pcd = np.min(pcd,axis=0)
        vt = len(pcd) / (max_pcd[0]-min_pcd[0])*(max_pcd[1]-min_pcd[1])*(max_pcd[2]-min_pcd[2])
        nonzero_indices = np.nonzero(desc)
        vi = desc[nonzero_indices]
        entropy = -np.sum((vi /vt) * np.log(vi /vt))

        return entropy
    
    
    def one_loop_cal_all_active_new(self,lidar_data):
        semantic_pt = np.array([list(elem) for elem in lidar_data])
        return self.active_920(semantic_pt)

    def build_vehicle2graph(self,vehicle_list):
        vehicle_graph = np.zeros((len(vehicle_list),len(vehicle_list)))
        for i in range(len(vehicle_list)):
            for j in range(len(vehicle_list)):
                if i != j:
                    vehicle_graph[i][j] = vehicle_list[i].get_transform().location.distance(vehicle_list[j].get_transform().location)
        return vehicle_graph
    

    def active_920(self, semantic_pt):
        # 1. make scan and bev desc
        scan_desc = np.zeros((self.PC_NUM_RING, self.PC_NUM_SECTOR))
        bev_max = np.zeros((self.PC_MAX_RANGE, self.PC_MAX_RANGE))
        bev_min = np.zeros((self.PC_MAX_RANGE, self.PC_MAX_RANGE))
        pt_range = self.PC_MAX_RANGE / 2

        # 2. select near points and build desc
        pt_x = semantic_pt[:, 0]
        pt_y = semantic_pt[:, 1]
        pt_z = semantic_pt[:, 2]

        azim_range = np.sqrt(pt_x ** 2+ pt_y ** 2)
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

        scan_entropy = self.scene_entropy(scan_desc, semantic_pt[:, :3])
        bev_entropy = self.scene_entropy(bev_scan, semantic_pt[:, :3])
        current_entropy_score = bev_entropy / scan_entropy

        vehicle_points, current_trans, label_output = {}, {}, []

        valid_vehicle_labels = np.isin(semantic_pt[:, 5], list(self.Vehicle_tags.keys()))
        valid_vehicle_points = semantic_pt[valid_vehicle_labels]
        unique_vehicle_labels = np.unique(valid_vehicle_points[:, 4])
        for label in unique_vehicle_labels:
            vehicle_points[int(label)] = valid_vehicle_points[valid_vehicle_points[:, 4] == label]
        
        actor_list = self.world.get_actors()
        filtered_actors = [actor for actor in actor_list if actor.id in vehicle_points.keys() and re.match("^vehicle", str(actor.type_id))]
        actor_dist = [actor.get_transform().location.distance(self.carla_actor.get_transform().location) for actor in actor_list]
        selected_actors = [actor for actor, dist in zip(filtered_actors, actor_dist) if dist < self.Largest_label_range]
        far_actors = [actor for actor, dist in zip(filtered_actors, actor_dist) if dist >= 0.7 * self.Largest_label_range and dist < self.Largest_label_range]
        
        walker_points = {}
        valid_walker_labels = np.isin(semantic_pt[:, 5], list(self.Walker_tags.keys()))
        valid_walker_points = semantic_pt[valid_walker_labels]
        unique_walker_labels = np.unique(valid_walker_points[:, 4])
        for label in unique_walker_labels:
            walker_points[int(label)] = valid_walker_points[valid_walker_points[:, 4] == label]
        walker_list = [actor for actor in actor_list if actor.id in walker_points.keys() and re.match("^walker", str(actor.type_id))]

        actor_cnt = len(selected_actors) - 0.5 * len(far_actors)

        # carla_actor_transform = self.carla_actor.get_transform().location
        # carla_actor_rotation_yaw = self.carla_actor.get_transform().rotation.yaw
        # for actor in selected_actors:
        #     bbox = actor.bounding_box
        #     actor_id = actor.id
        #     points_collection = actor_points[actor_id]

        #     max_p = np.max(points_collection, axis=0)
        #     min_p = np.min(points_collection, axis=0)
        #     cx = (max_p[0] + min_p[0]) / 2
        #     cy = (max_p[1] + min_p[1]) / 2
        #     cz = (actor.get_transform().location.z - carla_actor_transform.z + bbox.location.z)
        #     sx = 2 * bbox.extent.x
        #     sy = 2 * bbox.extent.y
        #     sz = 2 * bbox.extent.z
        #     yaw = (actor.get_transform().rotation.yaw - carla_actor_rotation_yaw + bbox.rotation.yaw)

        #     label_str = "{} {} {} {} {} {} {} {}".format(cx, cy, cz, sx, sy, sz, yaw, self.Label_dict[actor.semantic_tags[0]])
        #     label_output.append(label_str)
 
        
        temp_str = "{} {} {} {}".format(actor_cnt,scan_entropy,bev_entropy,current_entropy_score)
        label_output.append(temp_str)

        return True, label_output, current_entropy_score
    