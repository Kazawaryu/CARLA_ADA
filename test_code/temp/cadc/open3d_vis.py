import open3d as o3d
import numpy as np
import os

dir_path = '/home/ghosnp/dataset/cadc/data/0008/labeled/lidar_points/data'
file_name = '0000000025.bin'
def load_lidar(dir_path, file_name):
    lidar_path = os.path.join(dir_path, file_name)
    scan_data = np.fromfile(lidar_path, dtype=np.float32)  # numpy from file reads binary file
    # scan_data is a single row of all the lidar values
    # 2D array where each row contains a point [x, y, z, intensity]
    # we covert scan_data to format said above
    lidar = scan_data.reshape((-1, 4))

    lidar_x = lidar[:, 0]
    lidar_y = lidar[:, 1]
    lidar_z = lidar[:, 2]
    lidar_intensity = lidar[:, 3]
    return lidar_x, lidar_y, lidar_z, lidar_intensity

def visualize_lidar(lidar_x, lidar_y, lidar_z, lidar_intensity):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([lidar_x, lidar_y, lidar_z], axis=1))
    pcd.colors = o3d.utility.Vector3dVector(np.stack([lidar_intensity, lidar_intensity, lidar_intensity], axis=1))
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    lidar_x, lidar_y, lidar_z, lidar_intensity = load_lidar(dir_path, file_name)
    visualize_lidar(lidar_x, lidar_y, lidar_z, lidar_intensity)

    