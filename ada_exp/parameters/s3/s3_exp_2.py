# This file is used to visualize the result of the s3 score
import open3d as o3d
import numpy as np
import pandas as pd
import math
import os
from matplotlib import pyplot as plt

import sklearn.cluster as cluster

def draw_plt(path_list):
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    points = []
    for [start_path, sub_dir] in path_list:
        for sub in sub_dir:
            score_file = os.path.join(start_path, sub, 's3_score.csv')
            score_data = pd.read_csv(score_file)

            # score_data = score_data[score_data['s3_rho']<100]
            # score_data = score_data[score_data['s3_rho']>0]
            points.append(score_data[['s3_rho', 'fo']])
            ax1.scatter(score_data['s3_rho'], score_data['fo'], label=sub)

    ax1.legend()
    ax1.set_xlabel('s3_rho')
    ax1.set_ylabel('average attenuation')
    ax1.set_title('s3 score')
    
    # points to float
    points = pd.concat(points)
    points = points.astype('float64')

    print(np.min(points['s3_rho']),np.min(points['fo']))
    points['fo'] = points['fo'] / points['fo'].max()
    points['s3_rho'] = points['s3_rho'] / points['s3_rho'].max()

    # Use a cluster model to cluster points[]
    cluster_model = cluster.Birch(threshold=0.1, branching_factor=50, n_clusters=None)
    cluster_model.fit(points)

    labels = cluster_model.labels_
    # Visualize the result of the cluster
    ax2.scatter(points['s3_rho'], points['fo'], c=labels)
    ax2.set_xlabel('weather point density')
    ax2.set_ylabel('average attenuation')
    ax2.set_title('cluster result')

    plt.show()

def show_3d(path_list):
    # visulize the metircs in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for [start_path, sub_dir] in path_list:
        for sub in sub_dir:
            score_file = os.path.join(start_path, sub, 's3_score.csv')
            score_data = pd.read_csv(score_file)
            score_data = score_data[score_data['s3_rho']<10]
            # score_data = score_data[score_data['var']/score_data['nr']<0.01]
            # score_data = score_data[score_data['s3_rho']>0]
            # score_data = score_data[score_data['var']/score_data['nr']<0.01]
            ax.scatter(score_data['fo'], score_data['s3_rho'], score_data['var']/score_data['nr'], label=sub)
    ax.legend()
    ax.set_xlabel('beta')
    ax.set_ylabel('alpha')
    ax.set_zlabel('var/mean')
    ax.set_title('s3 score')
    plt.show()

    return

if __name__ == '__main__':
    path_list = []
    start_path = '/home/ghosnp/dataset/cadc/data/'
    sub_dir = ['0001', '0002', '0005', '0006', '0008', '0009']
    path_list.append([start_path, sub_dir])

    # start_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1024_1907/'
    # sub_dir =['vehicle.tesla.model3.master/']
    # path_list.append([start_path, sub_dir])

    # start_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1024_1907/'
    # sub_dir =['vehicle.tesla.model3.master/test/']
    # path_list.append([start_path, sub_dir])

    # start_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1020_2034/'
    # sub_dir =['vehicle.tesla.model3.master/']
    # path_list.append([start_path, sub_dir])

    # start_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1019_2027/'
    # sub_dir =['vehicle.tesla.model3.master/']
    # path_list.append([start_path, sub_dir])

    # start_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1019_2027/vehicle.tesla.model3.master/'
    # sub_dir =['velodyne_CVL_beta_0.002/s3_score/','velodyne_CVL_beta_0.005/s3_score/','velodyne_CVL_beta_0.008/s3_score/',
    #           'velodyne_CVL_beta_0.010/s3_score/','velodyne_CVL_beta_0.013/s3_score/','velodyne_CVL_beta_0.017/s3_score/']
    # path_list.append([start_path, sub_dir])
    # draw_plt(path_list)
    show_3d(path_list)