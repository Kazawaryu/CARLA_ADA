# This file is used to calculate the s3 score of the dataset

import open3d as o3d
import numpy as np
import math
import os
import time

import sklearn.cluster as cluster
from sklearn import metrics
from matplotlib import pyplot as plt

class s3_score_tool():
    def __init__(self) -> None:
        super().__init__()
        self.RADUIS = 18
        self.DEPTH = -0.2
        self.RANGE = 5
        self.K = 0.5
        self.DB = cluster.DBSCAN(eps=1.5, min_samples=15)

    def print_table(self, header, data):
        print('=' * 60)
        print('{:<20s}{:<20s}{:<20s}'.format(header[0], header[1], header[2]))
        print('=' * 60)
        for row in data:
            print('{:<20s}{:<20f}{:<20f}'.format(row[0], row[1], row[2]))
        print('=' * 60)

    def iter_get_alpha_beta(self, pcd, loop):
        # select probable points
        select_point = pcd[np.sqrt(pcd[:,0]**2 + pcd[:,1]**2) < self.RADUIS]
        select_point = select_point[select_point[:,2] > self.DEPTH]
        theta = np.arctan2(select_point[:,1], select_point[:,0])
        r = np.sqrt(select_point[:,0]**2 + select_point[:,1]**2)
        theta = theta*180/math.pi
        select_point_polar = np.array([theta, r]).T

        for _ in range(loop):
            if len(select_point_polar) == 0:
                select_point_polar = last_point_polar
                break
            self.DB.fit(select_point_polar[:,0:2])
            cls_labels = self.DB.labels_
            n_clusters_ = len(set(cls_labels)) - (1 if -1 in cls_labels else 0)
            label_point = {i: select_point_polar[cls_labels==i] for i in range(n_clusters_)}
            label_range = {i: [np.min(label_point[i][:,0]), np.max(label_point[i][:,0])] for i in range(n_clusters_)}
            label_range = sorted(label_range.items(), key=lambda x:x[1][0])

            # check if the cluster is a obstacle, use near points
            near_points = {-1:0}
            thetas,rs = select_point_polar[:,0], select_point_polar[:,1]
            for [k,v] in label_range:
                mask = np.logical_and(thetas>=v[0], thetas<=v[1])
                near_points[k] = np.sum(rs[mask]<self.RANGE)
                cls_labels[mask] = k
            near_points[-1] = len(select_point_polar) - sum(near_points.values())

            if len(near_points) == 1:
                break
            label_rho = {k: v/(label_range[k][1][1]-label_range[k][1][0]) for k,v in near_points.items()}
            label_length = {k: label_range[k][1][1]-label_range[k][1][0] for k in label_rho.keys()}
            max_item = max(label_length.items(), key=lambda x:x[1])
            origin_length = 360 - sum(label_length.values())
            if origin_length < max_item[1]:
                origin_length = max_item[1]
                label_rho.pop(max_item[0])
                label_length.pop(max_item[0])
                near_points[-1] = near_points[max_item[0]]

            base_rho = self.K * near_points[-1]/origin_length
            drop_key = [k for k, v in label_rho.items() if v < base_rho]

            # drop the points in the obstacle cluster
            indice = [i for i in range(select_point_polar.shape[0]) if cls_labels[i] not in drop_key]
            last_point_polar = select_point_polar
            select_point_polar = select_point_polar[indice]

        # transform polar to cartesian
        result_point = np.zeros((select_point_polar.shape[0], 3))
        result_point[:, 0] = select_point_polar[:, 1] * np.cos(select_point_polar[:, 0] * np.pi / 180)
        result_point[:, 1] = select_point_polar[:, 1] * np.sin(select_point_polar[:, 0] * np.pi / 180)
        result_point = np.array(result_point)

        label_range = {i: [np.min(label_point[i][:,0]), np.max(label_point[i][:,0])] for i in range(n_clusters_)}
        actual_len = sum([v[1]-v[0] for k,v in label_range.items()]) if len(label_range) > 0 else 360
        actual_rho = len(result_point) / actual_len

        other_points = pcd[np.sqrt(pcd[:,0]**2 + pcd[:,1]**2) >= self.RADUIS]
        indice = [i for i in range(select_point_polar.shape[0]) if cls_labels[i] in drop_key]
        other_points = np.concatenate((other_points, select_point[indice]), axis=0)
        other_points = other_points[other_points[:,3]!=0]
        all_points = pcd[pcd[:,3]!=0]

        sum_itensity_other = np.sum(other_points[:,3])
        num_other = other_points.shape[0]
        sum_itensity_all = np.sum(all_points[:,3])
        num_all = all_points.shape[0]
        num_result = result_point.shape[0]
        sum_itensity_result = sum_itensity_all-sum_itensity_other

        fo = self.get_beta(other_points[other_points[:,3]!=0])
        fa = self.get_beta(pcd[pcd[:,3]!=0])

        var = self.get_variance(select_point)

        return actual_rho, sum_itensity_result, num_result, sum_itensity_other, num_other, sum_itensity_all, num_all, fo, fa, var






    def get_s3_score(self,pcd):
        # select probable points
        select_point = pcd[np.sqrt(pcd[:,0]**2 + pcd[:,1]**2) < self.RADUIS]
        select_point = select_point[select_point[:,2] > self.DEPTH]
        theta = np.arctan2(select_point[:,1], select_point[:,0])
        r = np.sqrt(select_point[:,0]**2 + select_point[:,1]**2)
        theta = theta*180/math.pi
        select_point_polar = np.array([theta, r]).T

        # use DBSCAN to cluster the probable obstacle points
        db = self.DB.fit(select_point_polar[:,0:2])
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)       

        label_point = {i: select_point_polar[labels==i] for i in range(n_clusters_)}
        label_range = {i: [np.min(label_point[i][:,0]), np.max(label_point[i][:,0])] for i in range(n_clusters_)}
        label_range = sorted(label_range.items(), key=lambda x:x[1][0])

        # check if the cluster is a obstacle, use near points
        near_points = {-1:0}
        thetas,rs = select_point_polar[:,0], select_point_polar[:,1]
        for [k,v] in label_range:
            mask = np.logical_and(thetas>=v[0], thetas<=v[1])
            near_points[k] = np.sum(rs[mask]<self.RANGE)
            labels[mask] = k
        near_points[-1] = len(select_point_polar) - sum(near_points.values())



        label_rho = {k: v/(label_range[k][1][1]-label_range[k][1][0]) for k,v in near_points.items()}
        label_length = {k: label_range[k][1][1]-label_range[k][1][0] for k in label_rho.keys()}

        max_item = max(label_length.items(), key=lambda x:x[1])
        origin_length = 360 - sum(label_length.values())
        if origin_length < max_item[1]:
            origin_length = max_item[1]
            label_rho.pop(max_item[0])
            label_length.pop(max_item[0])
            near_points[-1] = near_points[max_item[0]]
        
        base_rho = self.K * near_points[-1]/origin_length
        drop_key = [k for k, v in label_rho.items() if v < base_rho] 

        # drop the points in the obstacle cluster
        indice = [i for i in range(select_point_polar.shape[0]) if labels[i] not in drop_key]
        result_point_polar = select_point_polar[indice]

        # transform polar to cartesian
        result_point = np.zeros((result_point_polar.shape[0], 3))
        result_point[:, 0] = result_point_polar[:, 1] * np.cos(result_point_polar[:, 0] * np.pi / 180)
        result_point[:, 1] = result_point_polar[:, 1] * np.sin(result_point_polar[:, 0] * np.pi / 180)
        result_point = np.array(result_point)

        actual_len = 360 - sum(label_range[k][1][1]-label_range[k][1][0] for k in drop_key)
        actual_rho = len(result_point) / actual_len

        other_points = pcd[np.sqrt(pcd[:,0]**2 + pcd[:,1]**2) >= self.RADUIS]
        indice = [i for i in range(select_point_polar.shape[0]) if labels[i] in drop_key]
        other_points = np.concatenate((other_points, select_point[indice]), axis=0)
        other_points = other_points[other_points[:,3]!=0]
        all_points = pcd[pcd[:,3]!=0]

        sum_itensity_other = np.sum(other_points[:,3])
        num_other = other_points.shape[0]
        sum_itensity_all = np.sum(all_points[:,3])
        num_all = all_points.shape[0]
        num_result = result_point.shape[0]
        sum_itensity_result = sum_itensity_all-sum_itensity_other


        fo = self.get_beta(other_points[other_points[:,3]!=0])
        fa = self.get_beta(pcd[pcd[:,3]!=0])

        var = self.get_variance(select_point)

        return actual_rho, sum_itensity_result, num_result, sum_itensity_other, num_other, sum_itensity_all, num_all, fo, fa, var
    
    def get_s3_score_from_file(self, dir_path ,file_path):
        pcd_path = os.path.join(dir_path, file_path)
        pre_point = np.fromfile(str(pcd_path), dtype=np.dtype([
                                    ('x', np.float32),
                                    ('y', np.float32),
                                    ('z', np.float32),
                                    ('intensity', np.float32),
                                ]) ,count=-1)

        pcd = np.array([list(elem) for elem in pre_point])

        return self.iter_get_alpha_beta(pcd, 3)
    
    def get_beta(self,cal_pcd):
        res = -np.log(cal_pcd[:,3]) / np.sqrt(cal_pcd[:,0]**2 + cal_pcd[:,1]**2 + cal_pcd[:,2]**2)
        aver_res = 100 * np.mean(res)
        return aver_res
    
    def get_variance(self, cal_pcd):
        grid_size = 0.2
        grid_num = int(2 * self.RADUIS / grid_size)
        grid_plane = np.zeros((grid_num, grid_num))
        place = np.floor(cal_pcd[:,0:2] / grid_size).astype(int)
        np.add.at(grid_plane, (place[:,0], place[:,1]), 1)
        var = np.var(grid_plane)
        return var

    
if __name__ == '__main__':
    start_path = '/home/ghosnp/dataset/cadc/data/'
    sub_dir = ['0001', '0002', '0005', '0006', '0008', '0009']
    next_path = 'labeled/lidar_points/data'

    # start_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1019_2027/'
    # sub_dir =['vehicle.tesla.model3.master/']
    # next_path = 'velodyne/'

    # start_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1020_2034/'
    # sub_dir =['vehicle.tesla.model3.master/']
    # next_path = 'velodyne_CVL_beta_0.019/'

    # start_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1019_2027/vehicle.tesla.model3.master/'
    # sub_dir =['velodyne_CVL_beta_0.002/','velodyne_CVL_beta_0.005/','velodyne_CVL_beta_0.008/',
    #           'velodyne_CVL_beta_0.010/','velodyne_CVL_beta_0.013/','velodyne_CVL_beta_0.017/']
    # next_path = ''


    score_tool = s3_score_tool()

    for sub in sub_dir:

        if next_path != '':
            save_path = os.path.join(start_path, sub)
        else:
            save_path = os.path.join(start_path, sub, 's3_score')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('save_path', save_path)
        if os.path.exists(os.path.join(save_path, 's3_score.csv')):
            os.remove(os.path.join(save_path, 's3_score.csv'))


        with open(os.path.join(save_path, 's3_score.csv'), 'w') as f:
            f.write('idx,s3_rho,ir,nr,io,no,ia,na,io/no,fo,fa,var\n')

        bin_path = os.path.join(start_path, sub, next_path)

        
        file_list = os.listdir(bin_path)
        file_list.sort()
        for file in file_list:
            if file.endswith('.bin'):
                tt = time.time()
                actual_rho, sum_itensity_result, num_result, sum_itensity_other, num_other, sum_itensity_all, num_all,fo,fa,var = score_tool.get_s3_score_from_file(bin_path, file)
                num_other = 1 if num_other == 0 else num_other

                with open(os.path.join(save_path, 's3_score.csv'), 'a') as f:
                    f.write(file[:-4]+','+str(actual_rho)+','+str(sum_itensity_result)+','+str(num_result)+','+str(sum_itensity_other)+','+str(num_other)+','+str(sum_itensity_all)+','+str(num_all)+','+str(sum_itensity_other/num_other)+','
                            +str(fo)+','+str(fa)+','+str(var)+'\n')
                print('dir',sub ,file[:-4], 'done! Cost:', time.time()-tt)