#!/usr/bin/python3

import re
import sys
import time
import open3d as o3d
import cv2
import math
import carla
import numpy as np

from recorder.sensor import Sensor
from active.lidar2 import ActiveLidar

class Lidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        # Save as a Nx4 numpy array. Each row is a point (x, y, z, intensity)
        lidar_data = np.fromstring(bytes(sensor_data.raw_data),
                                   dtype=np.float32)
        lidar_data = np.reshape(
            lidar_data, (int(lidar_data.shape[0] / 4), 4))

        # Convert point cloud to right-hand coordinate system
        lidar_data[:, 1] *= -1

        with open("{}/{:0>10d}.bin".format(save_dir,sensor_data.frame), 'wb') as file:
            file.write(lidar_data)

        # np.save("{}/{:0>10d}".format(save_dir,sensor_data.frame),lidar_data)
        return True


class SemanticLidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        self.carla_actor = carla_actor

    def set_world(self, world):
        self.active_lidar = ActiveLidar(world, self.carla_actor)


    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        # Save data as a Nx6 numpy array.
        lidar_data = np.fromstring(bytes(sensor_data.raw_data),
                                   dtype=np.dtype([
                                       ('x', np.float32),
                                       ('y', np.float32),
                                       ('z', np.float32),
                                       ('CosAngle', np.float32),
                                       ('ObjIdx', np.uint32),
                                       ('ObjTag', np.uint32)
                                   ]))

        # Convert point cloud to right-hand coordinate system
        lidar_data['y'] *= -1
        
        status, labels, cost = self.active_lidar.one_loop_cal_all_active_new(lidar_data)
        with open("{}/{:0>10d}.bin".format(save_dir,sensor_data.frame), 'wb') as file:
            file.write(lidar_data)

        with open("{}/{:0>10d}.txt".format(save_dir,sensor_data.frame),'a+',encoding='utf-8') as f:       
            for line in labels:
                print(line,file=f)

        return True
