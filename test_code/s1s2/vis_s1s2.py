import matplotlib.pyplot as plt
import numpy as np
import os

# time_dirc = '1128_2256'
# sem_pt_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'+time_dirc+'/vehicle.tesla.model3.master/velodyne_semantic/'

# time_dirc_list = ['1128_1836','1128_2219','1128_2256']
# labels_list = ['100-50','150-50','75-50']
time_dirc_list = ['1129_2035']
labels_list = ['75-50']

scan_entropy_list, bev_entropy_list, pf_scalar_list, pf_vector_list = [], [], [], []

for time_dirc in time_dirc_list:
    sem_pt_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'+time_dirc+'/vehicle.tesla.model3.master/velodyne_semantic/'
    file_list = os.listdir(sem_pt_path)
    file_list = [f for f in file_list if f.endswith('.txt')]
    file_list.sort()

    scan_entropy, bev_entropy, pf_scalar, pf_vector = [], [], [], []

    for file in file_list:
        with open(sem_pt_path+file, 'r') as f:
            s1s2 = f.readlines()[-1]
            s1s2 = s1s2.split(' ')
            scan_entropy.append(float(s1s2[1]))
            bev_entropy.append(float(s1s2[2]))
            pf_scalar.append(float(s1s2[3]))
            pf_vector.append(float(s1s2[4]))

    scan_entropy_list.append(scan_entropy)
    bev_entropy_list.append(bev_entropy)
    pf_scalar_list.append(pf_scalar)
    pf_vector_list.append(pf_vector)

# visualize the s1s2 in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(time_dirc_list)):
    ax.scatter(scan_entropy_list[i], bev_entropy_list[i], pf_scalar_list[i], label=labels_list[i])

ax.set_xlabel('scan_entropy')
ax.set_ylabel('bev_entropy')
ax.set_zlabel('pf_scalar')
ax.legend()
plt.show()
