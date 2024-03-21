import os
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
 
 
nusc = NuScenes(version='v1.0-mini', dataroot='/home/newDisk/nuscene', verbose=True)
my_scene = nusc.scene[0]
 
first_sample_token = my_scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)
 
my_annotation_token = sample['anns'][18]
my_annotation_metadata = nusc.get('sample_annotation', my_annotation_token)
nusc.render_annotation(my_annotation_token)
 
plt.show()
 
 
top_lidar_token = sample['data']['LIDAR_TOP']
top_lidar_data = nusc.get('sample_data', top_lidar_token)
 
pcd_bin_file = os.path.join(nusc.dataroot, top_lidar_data['filename'])
 
# Load the .pcd.bin file.
pc = LidarPointCloud.from_file(pcd_bin_file)
pcd = pc.points.T
pcd = pcd.reshape((-1, 4))[:, 0:3]
 
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pcd)
 
# 可视化点云
o3d.visualization.draw_geometries([point_cloud])
 
print("done")