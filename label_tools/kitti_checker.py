import numpy as np
import seaborn as sns
import mayavi.mlab as mlab
import argparse

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Bus', 'Truck', 'Pedestrian', 'Rider', 'Cyclist', 'Van', 'Misc', 'DontCare']

def read_args():
    parser = argparse.ArgumentParser(description='Plot 3D bounding box')
    parser.add_argument('--time_dir', '-t', type=str, help='recorder time directory')
    parser.add_argument('--data', '-d', type=str, help='data time stamp')
    args = parser.parse_args()
    bin_path = args.time_dir
    label_path = args.data
    
    return bin_path, label_path

if __name__ == '__main__':
    bin_path, label_path = read_args()
    # bin_path = '1026_2037'
    # label_path = '0000008601'
    # load point clouds
    scan_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'+bin_path+'/vehicle.tesla.model3.master/velodyne/' + label_path + '.bin'
    #scan_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/dataset/testing/velodyne/008571.bin'
    try:
        scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)
    except:
        scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 6)
    # load labels
    label_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'+bin_path+'/vehicle.tesla.model3.master/velodyne_semantic/' + label_path + '.txt'
    #label_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/dataset/testing/label_2/008571.txt'
    with open(label_dir, 'r') as f:
        labels = f.readlines()
    labels = labels[:-1]

    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
    # draw point cloud
    plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig)

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


        def draw(p1, p2, front=1):
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=colors[names.index(lab) * 2 + front], tube_radius=None, line_width=2, figure=fig)


        # draw the upper 4 horizontal lines
        draw(corners_3d[0], corners_3d[1], 0)  # front = 0 for the front lines
        draw(corners_3d[1], corners_3d[2])
        draw(corners_3d[2], corners_3d[3])
        draw(corners_3d[3], corners_3d[0])

        # draw the lower 4 horizontal lines
        draw(corners_3d[4], corners_3d[5], 0)
        draw(corners_3d[5], corners_3d[6])
        draw(corners_3d[6], corners_3d[7])
        draw(corners_3d[7], corners_3d[4])

        # draw the 4 vertical lines
        draw(corners_3d[4], corners_3d[0], 0)
        draw(corners_3d[5], corners_3d[1], 0)
        draw(corners_3d[6], corners_3d[2])
        draw(corners_3d[7], corners_3d[3])

    mlab.view(azimuth=230, distance=50)
    mlab.show()
