import open3d as o3d
import cv2
import sys
import numpy as np
import argparse
import math
from pathlib import Path

# try:
#     path = "/home/ghosnp/carla/usable_version_tool/uv2/carla_dataset_tools/utils/patchwork-plusplus/python_wrapper"
#     sys.path.insert(0, path)
#     import pypatchworkpp as ppp
# except:
#     print(sys.path)
#     exit(1)

offset_degree = 0


def config_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='1729')
    parser.add_argument('-s', type=str, default='0000404165')
    args = parser.parse_args()


    dir = args.d
    spec = args.s

    pcd = "/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_"+dir+"/vehicle.tesla.model3.master/velodyne/"+spec+".bin"
    txt = "/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_"+dir+"/vehicle.tesla.model3.master/velodyne_semantic/"+spec+".txt"



    return pcd,txt


def main():
    pcd_path, txt_path = config_path()
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
    

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 800, height = 600)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:,:3])

    vis.add_geometry(mesh)
    vis.add_geometry(pcd_o3d)

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


    pred_box = load_pre_label(txt_path) 
    exp_draw_boxes = get_draw_box(pred_box)
    for box in exp_draw_boxes:
        vis.add_geometry(box)

    render_option = vis.get_render_option()
    render_option.point_size = 2
    render_option.background_color = np.asarray([0, 0, 0])


    vis.run()  
    vis.destroy_window()


    return


def rotz(t):
    c = np.cos(t - offset_degree)
    s = np.sin(t - offset_degree)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def box2corner(box):
    x = box[0]
    y = box[1]
    z = box[2]
    l = box[3]  # dx
    w = box[4]  # dy
    h = box[5]  # dz
    yaw = box[6]
    Box = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )

    R = rotz(yaw)
    corners_3d = np.dot(R, Box)  # corners_3d: (3, 8)
    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    return np.transpose(corners_3d)


def get_line_set(corners):
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],]
    colors = [[1,0,0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def load_pre_label(gt_json_path):
    if Path(gt_json_path).exists():
        pred_box = []
        with open(gt_json_path, 'r') as fobj:

            for line in fobj:
                l = line.strip().split(" ")
                try:
                    cx, cy, cz, sx, sy, sz, yaw, label = l[0], l[1],l[2], l[3],l[4], l[5],l[6], l[7]
                    box_data = list(map(float,[ cx, cy, cz, sx, sy, sz, yaw]))
                    pred_box.append(box_data)
                    # print("[label_str]",line)
                except:
                    print("[entropy infos]",line)
                
        return pred_box

def get_draw_box(pre_box_set):
    draw_boxes = []
    for box_p in pre_box_set:
        cx, cy, cz, sx, sy, sz, yaw = box_p[0],box_p[1],box_p[2], box_p[3],box_p[4], box_p[5],box_p[6]
        corner_box = box2corner([cx, cy, cz, sx, sy, sz, yaw])
        draw_box = get_line_set(corner_box)
        draw_boxes.append(draw_box)
    return draw_boxes

if __name__ == "__main__":
    main()