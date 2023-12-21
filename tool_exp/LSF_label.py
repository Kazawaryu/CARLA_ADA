import os
import numpy as np
import open3d as o3d
import util

bin_path = '1221_1713'
label_path = '0000005738'
scan_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'+bin_path+'/vehicle.tesla.model3.master/velodyne/' + label_path + '.bin'
label_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'+bin_path+'/vehicle.tesla.model3.master/velodyne_semantic/' + label_path + '.txt'
gt_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'+bin_path+'/vehicle.tesla.model3.master/velodyne_semantic/' + label_path + '.bin'
usable_labels = {12.,14.,15.,16.,19.}
label_dict = {12.:'Pedestrian',14.:'Car',15.:'Truck',16.:'Bus',18:"Motorcycle",19.:'Bicycle'}

ret_tool = util.LShapeFitting()

def read_pcd():
    point = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)

    return point

def read_label_yaw():
    with open(label_dir, 'r') as f:
        labels = f.readlines()
    labels = labels[:-1]

    yaw_dict ={}
    h_dict = {}
    for line in labels:
        line = line.split()
        x, y, z, l, w, h, rot, lab, _, id = line
        yaw_dict[str(int(id))] = rot
        h_dict[str(int(id))] = h

    return yaw_dict, h_dict

def read_gt():
    semantic_points = np.fromfile(gt_dir, dtype=np.dtype([
                                       ('x', np.float32),
                                       ('y', np.float32),
                                       ('z', np.float32),
                                       ('CosAngle', np.float32),
                                        ('ObjIdx', np.uint32),
                                       ('ObjTag', np.uint32)
                                   ]) ,count=-1)
    semantic_points = np.array([list(elem) for elem in semantic_points])
    return semantic_points

def split_object(semantic_points):
    ground = []
    nonground = []
    objects = []
    objects_dict = {}

    for point in semantic_points:
        if point[5] in usable_labels:
            objects.append([point[0],point[1],point[2]])
            if not point[5] in objects_dict:
                objects_dict[point[5]] = {}
            if not point[4] in objects_dict[point[5]]:
                objects_dict[point[5]][point[4]] = []
            objects_dict[point[5]][point[4]].append(point)
        elif point[5] == 1.:
            ground.append([point[0],point[1],point[2]])
        else:
            nonground.append([point[0],point[1],point[2]])

    print("==========================================")
    get_semantic_prelabel(semantic_points)
    return ground,nonground,objects,objects_dict

def get_semantic_prelabel(semantic_pt):
    vehicle_points = {}
    valid_vehicle_labels = np.isin(semantic_pt[:, 5], list({14.:'Car',15.:'Truck',16.:'Bus'}))
    valid_vehicle_points = semantic_pt[valid_vehicle_labels]
    unique_vehicle_labels = np.unique(valid_vehicle_points[:, 4])
    for label in unique_vehicle_labels:
        vehicle_points[int(label)] = valid_vehicle_points[valid_vehicle_points[:, 4] == label]
    
    print("[per label]",len(vehicle_points))
    return vehicle_points

def get_object_corner(objects_dict, yaw_dict, h_dict):
    save_info = []
    corner_set={}


    for label, points in objects_dict.items():
        for oid, object_points in points.items():
            if str(int(oid)) not in yaw_dict.keys():
                continue
            else:    
                p_2d = np.array(object_points)[:,:2]
                yaw = yaw_dict[str(int(oid))]
                h = h_dict[str(int(oid))]
                print("yaw",yaw,"id", oid)
                max_z = np.max(object_points, axis=0)[2]
                min_z = np.min(object_points, axis=0)[2]
                corner_point = ret_tool.get_rectangle_given_theta(p_2d, yaw).calc_rect_contour()

                corners = np.array([[corner_point[0][0], corner_point[1][0], min_z],
                                        [corner_point[0][0], corner_point[1][0], max_z],
                                        [corner_point[0][1], corner_point[1][1], min_z],
                                        [corner_point[0][1], corner_point[1][1], max_z],
                                        [corner_point[0][2], corner_point[1][2], min_z],
                                        [corner_point[0][2], corner_point[1][2], max_z],
                                        [corner_point[0][3], corner_point[1][3], min_z],
                                        [corner_point[0][3], corner_point[1][3], max_z]])
                center = [(corners[4][0]+corners[0][0])/2,(corners[4][1]+corners[0][1])/2,(corners[4][2]+corners[0][2])/2]
                l = np.sqrt((corners[2][0] - corners[0][0])**2 + (corners[2][1] - corners[0][1])**2)
                w = np.sqrt((corners[6][0] - corners[0][0])**2 + (corners[6][1] - corners[0][1])**2)
                label_str = "{} {} {} {} {} {} {} {} {}" .format(center[0], center[1], center[2],
                                                                l, w, h,
                                                                yaw, label_dict[label], oid)
                save_info.append(label_str)
                if not label in corner_set:
                    corner_set[label] = {}
                corner_set[label][oid] = corners

    return save_info, corner_set


def open3d_draw_picture(ground,nonground,dect_objects,corner_set):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 800, height = 600)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    ground_o3d = o3d.geometry.PointCloud()
    ground_o3d.points = o3d.utility.Vector3dVector(ground)
    ground_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.8, 0.2] for _ in range(len(ground))])
    )

    nonground_o3d = o3d.geometry.PointCloud()
    nonground_o3d.points = o3d.utility.Vector3dVector(nonground)

    object_o3d = o3d.geometry.PointCloud()
    object_o3d.points = o3d.utility.Vector3dVector(dect_objects)
    object_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0] for _ in range(len(dect_objects))])
    )

    # alpha = 1.5
    # tick = time.time()
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(object_o3d, alpha)   
    # print("cost",time.time()-tick)
    
    

    lines_box = np.array([[0,1],[2,3],[4,5],[6,7],[0,2],[6,4],[1,3],[7,5],[0,6],[2,4],[1,7],[3,5]])
    colors = np.array([[0, 1, 1] for j in range(len(lines_box))])
    box_set = []
    for label, objects in corner_set.items():
        for oid, corners in objects.items():
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines_box)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            box_set.append(line_set)
    for line_set in box_set:
        vis.add_geometry(line_set)

    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    # mesh_sphere.compute_vertex_normals()
    # mesh_sphere.scale(25.0, center=np.array([0, 0, 0]))
    # ball_line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_sphere)
    # ball_line_set.paint_uniform_color([0, 0, 1])
    # vis.add_geometry(ball_line_set)

    vis.add_geometry(mesh)
    vis.add_geometry(ground_o3d)
    vis.add_geometry(nonground_o3d)
    vis.add_geometry(object_o3d)

    render_option = vis.get_render_option()
    render_option.point_size = 2
    render_option.background_color = np.asarray([0, 0, 0])

    vis.run()
    vis.destroy_window()


def main():
    semantic_point = read_gt()
    ground,nonground,objects,objects_dict = split_object(semantic_point)
    yaw_dict, h_dict = read_label_yaw()

    _, corner_set = get_object_corner(objects_dict, yaw_dict, h_dict)

    open3d_draw_picture(ground,nonground,objects,corner_set)

    return

if __name__ == '__main__':
    main()