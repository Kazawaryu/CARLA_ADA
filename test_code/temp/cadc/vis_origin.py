import open3d as o3d
import cv2
import sys
import numpy as np
import argparse
import math
import os
from pathlib import Path

# try:
#     path = "/home/ghosnp/carla/usable_version_tool/uv2/carla_dataset_tools/utils/patchwork-plusplus/python_wrapper"
#     sys.path.insert(0, path)
#     import pypatchworkpp as ppp
# except:
#     print(sys.path)
#     exit(1)

def read_parse_args():
    parser = argparse.ArgumentParser(description='Visualize the lidar point cloud')
    parser.add_argument('--dir', '-d',type=str, default='2')
    parser.add_argument('--bin', '-b',type=str, default='25')
    args = parser.parse_args()
    return args


def main():
    args = read_parse_args()
    sub_cnt, bin_cnt = args.dir, args.bin
    dir_path = '/home/ghosnp/dataset/cadc/data/000'+sub_cnt+'/labeled/lidar_points/data'
    file_name = '00000000'+bin_cnt+'.bin'

    dir_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1024_1907/vehicle.tesla.model3.master/velodyne'
    file_name = '0000004946.bin'

    pcd_path = os.path.join(dir_path, file_name)
    pre_point = np.fromfile(str(pcd_path), dtype=np.dtype([
                                       ('x', np.float32),
                                       ('y', np.float32),
                                       ('z', np.float32),
                                       ('intensity', np.float32),
                                   ]) ,count=-1)

    pcd = np.array([list(elem) for elem in pre_point])

    # ground = ppp.getGround(pcd)
    # nonground = ppp.getNonground(pcd)
    # time_taken = ppp.getTimeTaken()

    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    # mesh_sphere.compute_vertex_normals()
    # mesh_sphere.scale(25.0, center=np.array([0, 0, 0]))
    # ball_line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_sphere)
    # ball_line_set.paint_uniform_color([0, 0, 1])
    # vis.add_geometry(ball_line_set)

    # get the point whose distance to the origin is smaller than radius, and height > 0
    radius = 18
    depth = -0.2

    pcd50 = pcd[np.sqrt(pcd[:,0]**2 + pcd[:,1]**2) < radius]
    pcd50 = pcd50[pcd50[:,2] > depth]
    pcd50_o3d = o3d.geometry.PointCloud()
    pcd50_o3d.points = o3d.utility.Vector3dVector(pcd50[:,:3])
    pcd50_o3d.paint_uniform_color([1, 0, 0])

    # pcdnew is the point which is not in pcd50
    pcdnew = pcd[np.sqrt(pcd[:,0]**2 + pcd[:,1]**2 ) >= radius]
    pcd50in = pcd[np.sqrt(pcd[:,0]**2 + pcd[:,1]**2 ) < radius]
    pcd50in = pcd50in[pcd50in[:,2] < depth]

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 800, height = 600)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcdnew[:,:3])
    pcd_50_in = o3d.geometry.PointCloud()
    pcd_50_in.points = o3d.utility.Vector3dVector(pcd50in[:,:3])

    vis.add_geometry(mesh)
    vis.add_geometry(pcd_o3d)
    vis.add_geometry(pcd50_o3d)
    vis.add_geometry(pcd_50_in)


    # ground_o3d = o3d.geometry.PointCloud()
    # ground_o3d.points = o3d.utility.Vector3dVector(ground[:,:3])
    # ground_o3d.paint_uniform_color([0, 1, 0])

    # nonground_o3d = o3d.geometry.PointCloud()
    # nonground_o3d.points = o3d.utility.Vector3dVector(nonground[:,:3])
    # nonground_o3d.paint_uniform_color([1, 0, 0])

    # draw a circle line, r=100
    # circle = o3d.geometry.TriangleMesh.create_sphere(radius=100)
    # circle.compute_vertex_normals()
    # circle.scale(1.0, center=np.array([0, 0, 0]))
    # circle_line_set = o3d.geometry.LineSet.create_from_triangle_mesh(circle)
    # circle_line_set.paint_uniform_color([0, 0, 1])
    # vis.add_geometry(circle_line_set)


    
    render_option = vis.get_render_option()
    render_option.point_size = 2
    render_option.background_color = np.asarray([0, 0, 0])


    vis.run()  
    vis.destroy_window()


    return

if __name__ == "__main__":
    main()