path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1018_2027/vehicle.tesla.model3.master/velodyne_semantic'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(source_dir):
    # only read the last line of the file
    with open(source_dir, "r") as file:
        data = file.readlines()[-1]
    return data


df = pd.DataFrame(columns=['actor_cnt', 'scan_entropy', 'bev_entropy', 'current_entropy_score', 'filtered_actors', 'selected_walkers', 'pf_sum', 'pf_tan', 'pf_res'])
for filename in os.listdir(path):
    if filename.endswith(".txt"):
        data = read_data(os.path.join(path, filename)).split(" ")
        df.loc[len(df)] = data


# change pf_sum, pf_res, actor_cnt type into float
df['pf_sum'] = df['pf_sum'].astype(float)
df['pf_res'] = df['pf_res'].astype(float)
df['actor_cnt'] = df['actor_cnt'].astype(float)


# draw the curve of (pf_sum,pf_res,actor_cnt) in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['pf_sum'], df['pf_res'], df['actor_cnt'])
ax.set_xlabel('pf_sum')
ax.set_ylabel('pf_res')
ax.set_zlabel('actor_cnt')
ax.set_title('(pf_sum, pf_res, actor_cnt) Curve')
plt.show()