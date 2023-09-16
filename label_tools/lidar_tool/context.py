import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import argparse
import time


# GOOD DATA: bev_entropy / scan_entropy < 0.4

class context:
    def init(self):
        self.PC_MAX_RANGE = 60
        self.PC_NUM_RING = 60
        self.PC_NUM_SECTOR = 60


    def main(self):
        pcd_path,script_mode = self.config_args()

        pre_pcd = np.fromfile(str(pcd_path), dtype=np.dtype([
                                        ('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('intensity', np.float32),
                                    ]) ,count=-1)
        pcd = np.array([list(elem) for elem in pre_pcd])

        if script_mode == 'e':
            t = time.time()
            # print("origin function:")
            # scan_desc,bev_scan = self.l1_context(pcd)
            # print("scan entropy:",self.scene_entropy(scan_desc,pcd))
            # print("bev entropy:",self.scene_entropy(bev_scan,pcd))
            # print("cost time:",time.time()-t)

            # t = time.time()
            # print("speed up function:")
            scan_desc,bev_scan = self.l1_context_spup(pcd)
            scan_entropy = self.scene_entropy(scan_desc,pcd)
            bev_entropy = self.scene_entropy(bev_scan,pcd)
            print("scan entropy:",scan_entropy)
            print("bev entropy:",bev_entropy)
            print("cost time:",time.time()-t)

            print("cal res:", bev_entropy / scan_entropy)
        elif script_mode == 'v':
            self.vis_pcd(pcd)
        elif script_mode == 'h':
            scan_desc,bev_scan = self.l1_context(pcd)
            self.vis_heatmap(scan_desc)
            


    def l1_context(self,pcd):
        scan_desc = np.zeros((self.PC_NUM_RING, self.PC_NUM_SECTOR))
        bev_max = np.zeros((self.PC_MAX_RANGE,self.PC_MAX_RANGE))
        bev_min = np.zeros((self.PC_MAX_RANGE,self.PC_MAX_RANGE))

        pt_range = self.PC_MAX_RANGE / 2

        for i in range(len(pcd)):
            pt_x = pcd[i][0]
            pt_y = pcd[i][1]
            pt_z = pcd[i][2]

            azim_range = np.sqrt(pt_x**2 + pt_y**2)
            azim_angle = np.rad2deg(np.arctan2(pt_y, pt_x))

            if azim_range < self.PC_MAX_RANGE:
                if azim_angle < 0:
                    azim_angle += 360
                azim_sector = int(np.floor(azim_angle / 360 * self.PC_NUM_SECTOR))
                azim_ring = int(np.floor(azim_range / self.PC_MAX_RANGE * self.PC_NUM_RING))

                scan_desc[azim_ring][azim_sector] += 1

            if pt_x < pt_range and pt_x > -pt_range and pt_y < pt_range and pt_y > -pt_range:
                pt_x += pt_range
                pt_y += pt_range

                if pt_z > bev_max[int(pt_x)][int(pt_y)]: bev_max[int(pt_x)][int(pt_y)] = pt_z
                if pt_z < bev_min[int(pt_x)][int(pt_y)]: bev_min[int(pt_x)][int(pt_y)] = pt_z
                
        bev_scan = np.subtract(bev_max,bev_min)

        return scan_desc,bev_scan
    

    def l1_context_spup(self, pcd):
        scan_desc = np.zeros((self.PC_NUM_RING, self.PC_NUM_SECTOR))
        bev_max = np.zeros((self.PC_MAX_RANGE, self.PC_MAX_RANGE))
        bev_min = np.zeros((self.PC_MAX_RANGE, self.PC_MAX_RANGE))

        pt_range = self.PC_MAX_RANGE / 2

        pt_x = pcd[:, 0]
        pt_y = pcd[:, 1]
        pt_z = pcd[:, 2]

        azim_range = np.sqrt(pt_x**2 + pt_y**2)
        azim_angle = np.rad2deg(np.arctan2(pt_y, pt_x))
        azim_angle[azim_angle < 0] += 360

        valid_indices = np.where(azim_range < self.PC_MAX_RANGE)
        azim_sector = np.floor(azim_angle[valid_indices] / 360 * self.PC_NUM_SECTOR).astype(int)
        azim_ring = np.floor(azim_range[valid_indices] / self.PC_MAX_RANGE * self.PC_NUM_RING).astype(int)

        np.add.at(scan_desc, (azim_ring, azim_sector), 1)

        valid_indices = np.where((pt_x < pt_range) & (pt_x > -pt_range) & (pt_y < pt_range) & (pt_y > -pt_range))
        pt_x_valid = pt_x[valid_indices] + pt_range
        pt_y_valid = pt_y[valid_indices] + pt_range
        pt_z_valid = pt_z[valid_indices]

        bev_max_indices = (pt_x_valid.astype(int), pt_y_valid.astype(int))
        np.maximum.at(bev_max, bev_max_indices, pt_z_valid)

        bev_min_indices = (pt_x_valid.astype(int), pt_y_valid.astype(int))
        np.minimum.at(bev_min, bev_min_indices, pt_z_valid)

        bev_scan = np.subtract(bev_max, bev_min)

        return scan_desc, bev_scan


    def scan_context(self, pcd):
        scan_desc = np.zeros((self.PC_NUM_RING, self.PC_NUM_SECTOR))

        for i in range(len(pcd)):
            pt_x = pcd[i][0]
            pt_y = pcd[i][1]

            azim_range = np.sqrt(pt_x**2 + pt_y**2)
            azim_angle = np.rad2deg(np.arctan2(pt_y, pt_x))

            if azim_range < self.PC_MAX_RANGE:
                if azim_angle < 0:
                    azim_angle += 360
                azim_sector = int(np.floor(azim_angle / 360 * self.PC_NUM_SECTOR))
                azim_ring = int(np.floor(azim_range / self.PC_MAX_RANGE * self.PC_NUM_RING))

                scan_desc[azim_ring][azim_sector] += 1


        return scan_desc

    def bev_context(self,pcd):
        bev_max = np.zeros((self.PC_MAX_RANGE,self.PC_MAX_RANGE))
        bev_min = np.zeros((self.PC_MAX_RANGE,self.PC_MAX_RANGE))

        pt_range = self.PC_MAX_RANGE / 2

        for i in range(len(pcd)):
            pt_x = pcd[i][0]
            pt_y = pcd[i][1]
            pt_z = pcd[i][2]

            if pt_x < pt_range and pt_x > -pt_range and pt_y < pt_range and pt_y > -pt_range:
                pt_x += pt_range
                pt_y += pt_range

                if pt_z > bev_max[int(pt_x)][int(pt_y)]: bev_max[int(pt_x)][int(pt_y)] = pt_z
                if pt_z < bev_min[int(pt_x)][int(pt_y)]: bev_min[int(pt_x)][int(pt_y)] = pt_z

        bev_scan = np.subtract(bev_max,bev_min)
        return bev_scan
    
    def scene_entropy(self,desc,pcd):
        max_pcd = np.max(pcd[:,:3],axis=0)
        min_pcd = np.min(pcd[:,:3],axis=0)
        vt = len(pcd) / (max_pcd[0]-min_pcd[0])*(max_pcd[1]-min_pcd[1])*(max_pcd[2]-min_pcd[2])
        nonzero_indices = np.nonzero(desc)
        vi = desc[nonzero_indices]
        entropy = -np.sum((vi /vt) * np.log(vi /vt))

        return entropy
    
    def config_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', type=str)
        parser.add_argument('-m',type=str,default='entropy')
        args = parser.parse_args()

        pcd_path = args.p
        mode = args.m

        return pcd_path,mode

    def vis_pcd(self,pcd):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:,:3])

        vis.add_geometry(pcd_o3d)
        # vis.add_geometry(mesh)

        render_option = vis.get_render_option()
        render_option.point_size = 2
        render_option.background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()

        return

    def vis_heatmap(self,heatmap):
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        return


if __name__ == '__main__':
    context = context()
    context.init()
    context.main()