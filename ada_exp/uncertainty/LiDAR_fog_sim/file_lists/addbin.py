import os

path = '/home/newDisk/tool/carla_dataset_tool/ada_exp/uncertainty/LiDAR_fog_sim/file_lists/KITTI.txt'
# add '.bin' to each line, and save to the same file
with open(path, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() + '.bin\n' for line in lines]

with open(path, 'w') as f:
    f.writelines(lines)


