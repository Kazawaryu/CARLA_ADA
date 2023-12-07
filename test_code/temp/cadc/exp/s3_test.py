import open3d as o3d
import numpy as np
import argparse
import math
import os
import time

import sklearn.cluster as cluster
from sklearn import metrics
from matplotlib import pyplot as plt

def print_table(header, data):
    print('=' * 60)
    print('{:<20s}{:<20s}{:<20s}'.format(header[0], header[1], header[2]))
    print('=' * 60)
    for row in data:
        print('{:<20s}{:<20f}{:<20f}'.format(row[0], row[1], row[2]))
    print('=' * 60)

sub_cnt, bin_cnt = '6', '99'
dir_path = '/home/ghosnp/dataset/cadc/data/000'+sub_cnt+'/labeled/lidar_points/data'
file_name = '00000000'+bin_cnt+'.bin'
pcd_path = os.path.join(dir_path, file_name)
pre_point = np.fromfile(str(pcd_path), dtype=np.dtype([
                                    ('x', np.float32),
                                    ('y', np.float32),
                                    ('z', np.float32),
                                    ('intensity', np.float32),
                                ]) ,count=-1)

pcd = np.array([list(elem) for elem in pre_point])
RADIUS = 18
DEPTH = -0.2
RANGE = 5
K = 0.5
######################################################
start_time = time.time()
# select probable points
select_point = pcd[np.sqrt(pcd[:,0]**2 + pcd[:,1]**2) < RADIUS]
select_point = select_point[select_point[:,2] > DEPTH]
theta = np.arctan2(select_point[:,1], select_point[:,0])
r = np.sqrt(select_point[:,0]**2 + select_point[:,1]**2)
theta = theta*180/math.pi
select_point_polar = np.array([theta, r]).T


# use DBSCAN to cluster the probable obstacle points
db = cluster.DBSCAN(eps=1.5, min_samples=15).fit(select_point_polar[:,0:2])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
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
    near_points[k] = np.sum(rs[mask]<RANGE)
    labels[mask] = k
near_points[-1] = len(select_point_polar) - sum(near_points.values())


label_rho = {k: v/(label_range[k][1][1]-label_range[k][1][0]) for k,v in near_points.items()}
label_length = sum(label_range[k][1][1]-label_range[k][1][0] for k in label_rho.keys())
base_rho = K * near_points[-1]/(360-label_length)
drop_key = [k for k, v in label_rho.items() if v < base_rho]

# drop the points in the obstacle cluster
indice = [i for i in range(select_point_polar.shape[0]) if labels[i] not in drop_key]
result_point_polar = select_point_polar[indice]

# transform polar to cartesian
result_point = np.zeros((result_point_polar.shape[0], 3))
result_point[:, 0] = result_point_polar[:, 1] * np.cos(result_point_polar[:, 0] * np.pi / 180)
result_point[:, 1] = result_point_polar[:, 1] * np.sin(result_point_polar[:, 0] * np.pi / 180)
result_point = np.array(result_point)
print('=' * 60)
print('time: ', time.time()-start_time)

actual_len = 360 - sum(label_range[k][1][1]-label_range[k][1][0] for k in drop_key)
actual_rho = len(result_point) / actual_len
print('actual_rho', actual_rho)
######################################################

other_points = pcd[np.sqrt(pcd[:,0]**2 + pcd[:,1]**2) >= RADIUS]
sum_itensity_other = np.sum(other_points[:,3])
num_other = other_points.shape[0]
sum_itensity_all = np.sum(pcd[:,3])
num_all = pcd.shape[0]
num_result = result_point.shape[0]
sum_itensity_result = sum_itensity_all-sum_itensity_other

num_pre = select_point.shape[0]
sum_itensity_pre = np.sum(select_point[:,3])

header = ['Points', 'sum_intensity', 'num_points']

data = [
    ['Result points', sum_itensity_result, num_result],
    ['Pre_select points', sum_itensity_pre, num_pre],
    ['Other points', sum_itensity_other, num_other],
    ['All points', sum_itensity_all, num_all]
]
print_table(header, data)

# plot the result
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14, 6))
ax1.scatter(select_point[:,0], select_point[:,1])
ax1.set_title('origin point')
ax2.scatter(result_point[:,0], result_point[:,1])
ax2.set_title('result point')
plt.show()