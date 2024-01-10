import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class exp_tool:
    def __init__(self):
        ADA_Range = 40


        self.Walker_tags = {12:"Pedestrian",13:"Rider"}
        self.Vehicle_tags ={14:"Car",15:"Truck",16:"Van"}
        self.Cyclist_tags = {18:"Motorcycle",19:"Bicycle"}
        self.Active = True
        self.Largest_label_range = 100       # object labeling range, max 100
        # ===== newest version (11.28) parameters =====
        # S1 parameters
        self.PC_MAX_RANGE = ADA_Range
        self.PC_NUM_RING = ADA_Range
        self.PC_NUM_SECTOR = ADA_Range
        self.PC_MIN_Z = -2.3
        self.PC_MAX_Z = 0.7


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
        # the max x and max y should be less than self.PC_MAX_RANGE
        bev_max_indices = (np.clip(bev_max_indices[0], 0, self.PC_MAX_RANGE - 1), np.clip(bev_max_indices[1], 0, self.PC_MAX_RANGE - 1))
        np.maximum.at(bev_max, bev_max_indices, pt_z_valid)

        bev_min_indices = (pt_x_valid.astype(int), pt_y_valid.astype(int))
        bev_min_indices = (np.clip(bev_min_indices[0], 0, self.PC_MAX_RANGE - 1), np.clip(bev_min_indices[1], 0, self.PC_MAX_RANGE - 1))
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
    
set_label = ['50-25','75-37','100-50','125-67','150-75']
set_A_05 = ['1226_2120', '1226_2132', '1226_2146', '1226_2200', '1226_2214']
set_B_02 = ['0104_2223', '0104_2309', '0104_2328', '0104_2343', '0105_0013']
set_D_06 = ['0104_1949', '0104_2002', '0104_2016', '0104_2032', '0104_2056']
time_dirc_list = set_B_02
labels_list = set_label

label_valid = {'Car', 'Truck', 'Bus'}

idx, scan_entropy_list, bev_entropy_list, pf_scalar_list, pf_vector_list, s1_gt =[], [], [], [], [], []

exp_tool = exp_tool()

for i in range(len(time_dirc_list)):
    time_dirc = time_dirc_list[i]
    sem_pt_path = '/home/newDisk/tool/carla_dataset_tool/raw_data/record_2024_'+time_dirc+'/vehicle.tesla.model3.master/velodyne_semantic/'
    file_list = os.listdir(sem_pt_path)
    file_list = [f for f in file_list if f.endswith('.bin')]
    file_list.sort()
    
    for file in file_list:
        sem_pt = np.fromfile(sem_pt_path+file, dtype=np.float32).reshape(-1, 6)
        
        scan_ep, bev_ep = exp_tool.get_S1_score(sem_pt)
        gt = 0
        with open(sem_pt_path+file[:-4]+'.txt') as f:
            lines = f.readlines()
            last_line = lines[-1]
            scores = last_line.split(' ')
            for line in lines[:-1]:
                line = line.split()
                x, y, z, l, w, h, rot, lab, _, _ = line
                x, y, z, l, w, h, rot = map(float, [x, y, z, l, w, h, rot])
                dist = np.sqrt(x**2+y**2+z**2)
                
                if dist < exp_tool.PC_MAX_RANGE:
                    gt += 1

            scan_entropy_list.append(scan_ep)
            bev_entropy_list.append(bev_ep)
            pf_scalar_list.append(scores[3])
            pf_vector_list.append(scores[4].replace('\n',''))
            s1_gt.append(gt)

        print(i ,file, gt)

df = pd.DataFrame({'scan_entropy':scan_entropy_list, 'bev_entropy':bev_entropy_list, 'pf_scalar':pf_scalar_list, 'pf_vector':pf_vector_list, 's1_gt':s1_gt})
df['scan_enp'] = scan_entropy_list
df['bev_enp'] = bev_entropy_list
df['pf_scalar'] = pf_scalar_list
df['pf_vector'] = pf_vector_list
df['s1_gt'] = s1_gt

# save_path = './s1s2_score_D50.csv'
save_path = '/home/newDisk/tool/carla_dataset_tool/ada_exp/parameters/s1/s1s2_score_B40.csv'

df.to_csv(save_path, index=False)
