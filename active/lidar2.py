import time
import open3d as o3d
import numpy as np
import active.util as util

class ActiveLidar:
    def __init__(self,world,sensor) -> None:
        self.world = world
        self.carla_actor = sensor

        self.Walker_tags = {12:"Pedestrian",13:"Rider"}
        self.Vehicle_tags ={14:"Car",15:"Truck",16:"Van"}
        self.Cyclist_tags = {18:"Motorcycle",19:"Bicycle"}
        self.Active = True
        self.Largest_label_range = 100       # object labeling range, max 100
        # ===== newest version (11.28) parameters =====
        # S1 parameters
        self.PC_MAX_RANGE = 60
        self.PC_NUM_RING = 60
        self.PC_NUM_SECTOR = 60
        self.PC_MIN_Z = -2.3
        self.PC_MAX_Z = 0.7

        # S2 parameters
        self.PF_MIN_RANGE = 10
        self.PF_MAX_RANGE = 60
        self.PF_O3D = o3d.geometry.PointCloud()
        self.PF_ALPHA = 1.5
        self.PF_K = 1.0
        self.PF_D = 1E5
        self.PF_H = 0.5
        self.RET_TOOL = util.LShapeFitting()

    def get_S1_score(self,semantic_pt):
        # 1. make scan and bev desc
        scan_desc = np.zeros((self.PC_NUM_RING, self.PC_NUM_SECTOR))
        bev_max = np.zeros((self.PC_MAX_RANGE, self.PC_MAX_RANGE))
        bev_min = np.zeros((self.PC_MAX_RANGE, self.PC_MAX_RANGE))
        pt_range = self.PC_MAX_RANGE / 2
        # 2. select near points and build desc
        pt_x = semantic_pt[:, 0]
        pt_y = semantic_pt[:, 1]
        pt_z = semantic_pt[:, 2]
        valid_indices = np.where((pt_z < self.PC_MAX_Z) & (pt_z > self.PC_MIN_Z))     
        pt_x = pt_x[valid_indices]
        pt_y = pt_y[valid_indices]
        pt_z = pt_z[valid_indices]
        # 2.1 build scan desc
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

        # 2.2. build bev scan
        bev_max_indices = (pt_x_valid.astype(int), pt_y_valid.astype(int))
        np.maximum.at(bev_max, bev_max_indices, pt_z_valid)

        bev_min_indices = (pt_x_valid.astype(int), pt_y_valid.astype(int))
        np.minimum.at(bev_min, bev_min_indices, pt_z_valid)

        bev_scan = np.subtract(bev_max, bev_min)

        # 3. calculate entropy
        scan_entropy = self.cal_scene_entropy(scan_desc, semantic_pt[:, :3])
        bev_entropy = self.cal_scene_entropy(bev_scan, semantic_pt[:, :3])
        
        return scan_entropy, bev_entropy
    
    def cal_scene_entropy(self, desc, pcd):
        max_pcd = np.max(pcd,axis=0)
        min_pcd = np.min(pcd,axis=0)
        vt = len(pcd) / (max_pcd[0]-min_pcd[0])*(max_pcd[1]-min_pcd[1])*(max_pcd[2]-min_pcd[2])
        nonzero_indices = np.nonzero(desc)
        vi = desc[nonzero_indices]
        entropy = -np.sum((vi /vt) * np.log(vi /vt))

        return entropy
    
    def get_S2_score(self,semantic_pt):
        actor_list = self.world.get_actors()
        vehicle_points, walker_points ,label_output = {}, {} ,[]

        valid_vehicle_labels = np.isin(semantic_pt[:, 5], list(self.Vehicle_tags.keys()))
        valid_vehicle_points = semantic_pt[valid_vehicle_labels]
        unique_vehicle_labels = np.unique(valid_vehicle_points[:, 4])
        for label in unique_vehicle_labels:
            vehicle_points[int(label)] = valid_vehicle_points[valid_vehicle_points[:, 4] == label]
        vehicle_actors = [actor for actor in actor_list if actor.id in vehicle_points.keys()]

        valid_walker_labels = np.isin(semantic_pt[:, 5], list(self.Walker_tags.keys()))
        valid_walker_points = semantic_pt[valid_walker_labels]
        unique_walker_labels = np.unique(valid_walker_points[:, 4])
        for label in unique_walker_labels:
            walker_points[int(label)] = valid_walker_points[valid_walker_points[:, 4] == label]
        walker_actors = [actor for actor in actor_list if actor.id in walker_points.keys()]

        carla_actor_location = self.carla_actor.get_transform().location
        carla_actor_rotation_yaw = self.carla_actor.get_transform().rotation.yaw

        lambdas_, dists_, bsize_, degree_ = [], [], [], []
        s1_gt_ = 0.

        fix_yaw = np.deg2rad(180 - carla_actor_rotation_yaw)
        fix_rot_matrix_2d = np.array([[np.cos(fix_yaw), -np.sin(fix_yaw)],
                                    [np.sin(fix_yaw), np.cos(fix_yaw)]])
        for actor in vehicle_actors:
            bbox = actor.bounding_box 
            # max_p = np.max(vehicle_points[actor.id][:,:3],axis=0)
            # min_p = np.min(vehicle_points[actor.id][:,:3],axis=0)
            
            
            # object_position = actor.get_transform().location - carla_actor_location + bbox.location
            object_position = actor.get_transform().location - carla_actor_location
        
            cx_, cy_, cz_ = object_position.x, object_position.y, (actor.get_transform().location.z - carla_actor_location.z + bbox.location.z)
            sx_, sy_, sz_ = 2*bbox.extent.x, 2*bbox.extent.y, 2*bbox.extent.z       
            # cx_ , cy_ , cz_ = (max_p[0]+min_p[0])/2, (max_p[1]+min_p[1])/2, (actor.get_transform().location.z - carla_actor_transform.z + bbox.location.z)
            yaw_ = (actor.get_transform().rotation.yaw - carla_actor_rotation_yaw + bbox.rotation.yaw) * np.pi / 180
            tag_ = self.Vehicle_tags[vehicle_points[actor.id][0][5]]


            
            _dist = np.sqrt(cx_**2 + cy_**2 + cz_**2)
            if _dist < self.PF_MAX_RANGE: s1_gt_ += 1.
            if _dist < self.PF_MIN_RANGE:
                _mes = 2000
                lambdas_.append(_mes)
                dists_.append(_dist)
                bsize_.append(sx_*sy_*sz_)
                degree_.append(np.arctan2(cy_, cx_))
                # s1_gt_ += 1
            elif _dist < self.PF_MAX_RANGE:
                self.PF_O3D.points = o3d.utility.Vector3dVector(vehicle_points[actor.id][:, :3])
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.PF_O3D, self.PF_ALPHA)
                _mes = min(2000, len(np.asarray(mesh.triangles)))
                lambdas_.append(_mes)
                dists_.append(_dist)
                bsize_.append(sx_*sy_*sz_)
                degree_.append(np.arctan2(cy_, cx_))
                # s1_gt_ += 1
            else: _mes = 0

            [cx_, cy_] = np.dot(fix_rot_matrix_2d, np.array([-cx_, -cy_]))
            # custom format: cx, cy, cz, sx, sy, sz, yaw, lab, mes
            # label_ = [cx_, cy_, cz_, sx_, sy_, sz_, yaw_, tag_, mes_]
            # kitti format: -cy, -cz-0.5*sz, cx, sy, sx, -sz, yaw, lab, mes
            # label_ = [-cy_, -cz_+0.5*sz_, -cx_, sy_, sx_, sz_, yaw_, tag_, _mes, actor.id]
            label_ = [cy_, -cz_+0.5*sz_, cx_, sy_, sx_, sz_, yaw_, tag_, _mes, actor.id]
            label_output.append(label_)

        for actor in walker_actors:    
            # max_p = np.max(walker_points[actor.id][:,:3],axis=0)
            # min_p = np.min(walker_points[actor.id][:,:3],axis=0)

            bbox = actor.bounding_box
            # object_position = actor.get_transform().location - carla_actor_location + bbox.location
            object_position = actor.get_transform().location - carla_actor_location
            
            cx_, cy_, cz_ = object_position.x, object_position.y, (actor.get_transform().location.z - carla_actor_location.z + bbox.location.z)
            sx_, sy_, sz_ = 2*bbox.extent.x, 2*bbox.extent.y, 2*bbox.extent.z   

            # sx_, sy_, sz_ = max_p[0]-min_p[0], max_p[1]-min_p[1], max_p[2]-min_p[2]
            # cx_ , cy_ , cz_ = (max_p[0]+min_p[0])/2, (max_p[1]+min_p[1])/2, (actor.get_transform().location.z - carla_actor_transform.z + bbox.location.z)
            yaw_ = (actor.get_transform().rotation.yaw - carla_actor_rotation_yaw + bbox.rotation.yaw) * np.pi / 180
            tag_ = self.Walker_tags[walker_points[actor.id][0][5]]
            _mes = 0
            # custom format: cx, cy, cz, sx, sy, sz, yaw, lab, mes
            # label_ = [cx_, cy_, cz_, sx_, sy_, sz_, yaw_, tag_, mes_]
            # kitti format: -cy, -cz-0.5*sz, cx, sy, sx, -sz, yaw, lab, mes
            # label_ = [-cy_, -cz_+0.5*sz_, cx_, sy_, sx_, sz_, yaw_, tag_, _mes, actor.id]
            
            cx_, cy_ = np.dot(fix_rot_matrix_2d, np.array([-cx_, -cy_]))
            label_ = [cy_, -cz_+0.5*sz_, cx_, sy_, sx_, sz_, yaw_, tag_, _mes, actor.id]
            label_output.append(label_)

        lambdas_, dists_, bsize_, degree_ = np.array(lambdas_), np.array(dists_), np.array(bsize_), np.array(degree_)
        pf_l = dists_ / self.PF_MAX_RANGE
        pf_score = (dists_**2 * lambdas_ / np.sqrt(bsize_)) / (self.PF_D * (pf_l * np.log(pf_l) + self.PF_H))
        pf_scalar = np.sum(pf_score)
        pf_degree = np.arctan2(np.sum(pf_score*np.cos(degree_)), np.sum(pf_score*np.sin(degree_)))

        return label_output, pf_scalar, pf_degree, s1_gt_

    def labels_to_string(self, s1_gt_, label_output, se, be, ps, pd):
        return_strs = []
        for label in label_output:
            label_str = "{} {} {} {} {} {} {} {} {} {}".format(label[0], label[1], label[2], label[3], label[4], label[5], label[6], label[7], label[8], label[9])
            return_strs.append(label_str)
        return_strs.append("{} {} {} {} {}".format(s1_gt_, se, be, ps, pd))
        return return_strs
    
    def one_loop_cal_all_active_new(self, lidar_data):
        start_t = time.time()
        semantic_pt = np.array([list(elem) for elem in lidar_data])
        scan_entropy, bev_entropy = self.get_S1_score(semantic_pt)
        labels, pf_scalar, pf_degree, s1_gt_ = self.get_S2_score(semantic_pt)
        label_output = self.labels_to_string(s1_gt_, labels, scan_entropy, bev_entropy, pf_scalar, pf_degree)
        end_t = time.time()
        print("=================", scan_entropy, bev_entropy, pf_scalar, pf_degree, "=================")
        return True, label_output, end_t-start_t
        

