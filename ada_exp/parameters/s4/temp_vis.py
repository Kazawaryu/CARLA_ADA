import os
import open3d as o3d
import numpy as np
import torch


data_path = '/home/ghosnp/dataset/mini_kitti/velodyne/selected/000011.bin'


# vis the point cloud
pcd = np.fromfile(data_path, dtype=np.float32).reshape(-1, 4)

# use open3d to vis the point cloud
txt_path = '/home/ghosnp/dataset/mini_kitti/label_2/selected/000011.txt'

with open(txt_path, 'r') as f:
    labels = f.readlines()

bboxs = []

for line in labels:
    line = line.split()
    try:
        x, y, z, l, w, h, rot, lab, _ = line
    except:
        lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
    h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
    if lab != 'DontCare':
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # transform the 3d bbox from object coordiante to camera_0 coordinate
        R = np.array([[np.cos(rot), 0, np.sin(rot)],
                    [0, 1, 0],
                    [-np.sin(rot), 0, np.cos(rot)]])
        corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

        # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
        corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])

        bboxs.append(corners_3d)


pcd = pcd[:, :3]

pcd = pcd[pcd[:,1]>0]
pcd = pcd[pcd[:,0]>0]

vis = o3d.visualization.Visualizer()
vis.create_window()

pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(pcd)

vis.add_geometry(pcd_o3d)


bboxs = np.array(bboxs)
for bbox in bboxs:
    bbox_o3d = o3d.geometry.LineSet()
    bbox_o3d.points = o3d.utility.Vector3dVector(bbox)
    bbox_o3d.lines = o3d.utility.Vector2iVector(np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]))
    bbox_o3d.colors = o3d.utility.Vector3dVector(np.array([[1,0,0] for i in range(12)]))
    vis.add_geometry(bbox_o3d)

vis.run()
vis.destroy_window()