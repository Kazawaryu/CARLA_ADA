import argparse
import glob
from pathlib import Path
import open3d
import torch
import kornia
import numpy as np
import torch
import time
import demo as demo

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# 3 tool class, PillarVFE, PFNLayer, PointPillarScatter
from pcdet.models.backbones_3d.vfe import pillar_vfe as VFE
from pcdet.models.backbones_2d.map_to_bev import pointpillar_scatter as Scatter

ckpt_path = '../self_ckpts/pointpillar_7728.pth'
cfg_path = './cfgs/kitti_models/pointpillar.yaml'
pcd_path = '/home/ghosnp/dataset/mini_kitti/velodyne/training/velodyne/000011.bin'
txt_path = '/home/ghosnp/dataset/mini_kitti/label_2/training/label_2/000011.txt'

class isPointInQuadrangle(object):

    def __int__(self):
        self.__isInQuadrangleFlag = False

    def cross_product(self, xp, yp, x1, y1, x2, y2):
        return (x2 - x1) * (yp - y1)-(y2 - y1) * (xp - x1)

    def compute_para(self, xp, yp, xa, ya, xb, yb, xc, yc, xd, yd):
        cross_product_ab = isPointInQuadrangle().cross_product(xp, yp, xa, ya, xb, yb)
        cross_product_bc = isPointInQuadrangle().cross_product(xp, yp, xb, yb, xc, yc)
        cross_product_cd = isPointInQuadrangle().cross_product(xp, yp, xc, yc, xd, yd)
        cross_product_da = isPointInQuadrangle().cross_product(xp, yp, xd, yd, xa, ya)
        return cross_product_ab,cross_product_bc,cross_product_cd,cross_product_da

    def is_in_rect(self, aa, bb, cc, dd):
        if (aa > 0 and bb > 0 and cc > 0 and dd > 0) or (aa < 0 and bb < 0 and cc < 0 and dd < 0):
            self.__isInQuadrangleFlag= True
        else:
            self.__isInQuadrangleFlag = False

        return self.__isInQuadrangleFlag

def trans_data_2_tensor(batch_dict):
    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.cuda()
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int()
        else:
            batch_dict[key] = torch.from_numpy(val).float()

def cal_PCC(a,b):
    # a,b are two vectors, both are numpy array(64,)
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_std = np.std(a)
    b_std = np.std(b)
    a = (a - a_mean) / a_std
    b = (b - b_mean) / b_std

    pcc = np.sum(a * b) / (64 - 1)
    return pcc

def cal_DIST(a,b):
    # a,b are two vectors, both are numpy array(64,)
    # a_mean = np.mean(a)
    # b_mean = np.mean(b)
    # a_std = np.std(a)
    # b_std = np.std(b)
    # a = (a - a_mean) / a_std
    # b = (b - b_mean) / b_std

    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist

#####################################################################



# 1. load config
logger = common_utils.create_logger()  
cfg_from_yaml_file(cfg_path, cfg)

demo_dataset = demo.DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=Path(pcd_path), ext='.bin', logger=logger)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)

# 2. set tools
voxel_size = torch.tensor([0.16, 0.16, 4])
pc_range = torch.tensor([-69.12, -39.68, -3., 69.12, 39.68, 1.])

with torch.no_grad():
    for idx, data_dict in enumerate(demo_dataset):
        data_dict = demo_dataset.collate_batch([data_dict])
        trans_data_2_tensor(data_dict)

model_cfg = model.model_cfg
num_point_features = model_cfg.VFE.NUM_FILTERS
num_point_features = 4

in_channels = 10
out_channels = 64
use_norm = True
last_layer = True
grid_size = [(pc_range[3] - pc_range[0]) / voxel_size[0], (pc_range[4] - pc_range[1]) / voxel_size[1], 1]
grid_size = torch.tensor(grid_size).int()

MAX_I, MAX_PCC = 0, 0




for i in range(1):
    common_utils.set_random_seed(79)
    print('====================', i, '====================')
    # 3. get pillar feature
    pillarVFE = VFE.PillarVFE(model_cfg=model_cfg.VFE, num_point_features=num_point_features, voxel_size=voxel_size, point_cloud_range=pc_range)
    pillarVFE.forward(data_dict)

    pillarScatter = Scatter.PointPillarScatter(model_cfg=model_cfg.MAP_TO_BEV, grid_size=grid_size)
    pillarScatter.forward(data_dict)

    # get sample pillar feature
    corners_bevs = []
    corners_3Ds = []
    POINT_CLOUD_RANGE = np.array([-69.12, -39.68, -3, 69.12, 39.68, 1], dtype=np.float32)
    with open(txt_path, 'r') as f:
        txt = f.readlines()


    for line in txt:
        line = line.split()
        lab, x, y, z, w, l, h, rot = line[0], line[11], line[12], line[13], line[9], line[10], line[8], line[14]
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
            corners_3Ds.append(corners_3d)
            corners_bevs.append(corners_3d[:4, [0, 1]])
            
    corners_bevs = np.array(corners_bevs)

    # corners_bevs add range offset
    corners_bevs[:, :, 0] += POINT_CLOUD_RANGE[3]
    corners_bevs[:, :, 1] += POINT_CLOUD_RANGE[4]
    corners_bevs /= 0.16


    feature_idx = []
    tool = isPointInQuadrangle()
    # Get the index in the range of corners_bevs
    for corners in corners_bevs:

        vector01 = corners[1] - corners[0]
        vector03 = corners[3] - corners[0]
        square_area = np.linalg.norm(vector01) * np.linalg.norm(vector03)
        max_x, min_x, max_y, min_y = np.max(corners[:,0]), np.min(corners[:,0]), np.max(corners[:,1]), np.min(corners[:,1])
        max_x, min_x, max_y, min_y = int(max_x), int(min_x), int(max_y), int(min_y)
        # get x,y with (min_x < x < max_x and min_y < y < max_y)
        x = np.arange(min_x, max_x)
        y = np.arange(min_y, max_y)
        temp_add = []
        for i in x:
            for j in y:
                a, b, c, d = tool.compute_para(i, j, corners[0][0], corners[0][1], corners[1][0], corners[1][1], corners[2][0], corners[2][1], corners[3][0], corners[3][1])
                res = tool.is_in_rect(a, b, c, d)
                if res:
                    feature_idx.append([i, j])
                    temp_add.append([i, j])
        
    feature_idx = np.array(feature_idx)

    # feature_idx load to gpu
    feature_idx_gpu = torch.from_numpy(feature_idx).cuda()
    # load the spatial_features
    spatial_features_gpu = data_dict['spatial_features'][0]
    # get the feature with feature_idx
    valid_features_gpu = spatial_features_gpu[:, feature_idx_gpu[:,1], feature_idx_gpu[:,0]]
    valid_features = valid_features_gpu.detach().numpy().T

    # 5. load model feature
    for name, param in model.named_parameters():
        # print(name, param.shape)
        if name == 'vfe.pfn_layers.0.linear.weight':
            vfe_pfn_weight = param.detach().cpu().numpy()
        elif name == 'vfe.pfn_layers.0.norm.weight':
            vfe_pfn_norm_weight = param.detach().cpu().numpy()
        elif name == 'vfe.pfn_layers.0.norm.bias':
            vfe_pfn_norm_bias = param.detach().cpu().numpy()
        else:
            pass

    # 5. calculate the pcc and dist
    pccs , dists = [], []
    new_valid_features = []
    for i in range(valid_features.shape[0]):
        if np.sum(valid_features[i]) != 0:
            new_valid_features.append(valid_features[i])

    for f in new_valid_features:
        norm_f = (f - np.mean(f)) / np.std(f)
        pcc = cal_PCC(norm_f, vfe_pfn_norm_weight)
        pccs.append(pcc)

        # for i in vfe_pfn_weight.T:
        #     pcc = cal_PCC(f, i)
        #     pccs.append(pcc)
        
    random_vector = np.random.rand(64)
    print('RAND',cal_PCC(random_vector, vfe_pfn_norm_weight), cal_DIST(random_vector, vfe_pfn_norm_weight))
    print('S4-VAILD')
    print('PCC', np.mean(pccs), np.std(pccs))
