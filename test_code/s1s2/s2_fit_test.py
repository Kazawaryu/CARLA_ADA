import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path_1 = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1019_1520/vehicle.tesla.model3.master/velodyne_semantic'
path_2 = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_1019_1922/vehicle.tesla.model3.master/velodyne_semantic'

df = pd.DataFrame(columns=['cx', 'cy', 'cz', 'sx', 'sy', 'sz', 'yaw', 'tag', 'mesh_cnt', 'dist'])

for filename in os.listdir(path_1):
    if filename.endswith(".txt"):
        data = pd.read_csv(os.path.join(path_1, filename), sep=" ", header=None)
        data.columns = ["cx", "cy", "cz", "sx", "sy", "sz", "yaw", "tag", "mesh_cnt", "dist"]
        data = data[:-1]
        row = pd.DataFrame(data.iloc[0]).T
        df = pd.concat([df, row], ignore_index=True)

for filename in os.listdir(path_2):
    if filename.endswith(".txt"):
        data = pd.read_csv(os.path.join(path_2, filename), sep=" ", header=None)
        data.columns = ["cx", "cy", "cz", "sx", "sy", "sz", "yaw", "tag", "mesh_cnt", "dist"]
        data = data[:-1]
        row = pd.DataFrame(data.iloc[0]).T
        df = pd.concat([df, row], ignore_index=True)

# spilt the df by 'tag', totally 3 sub df, tag='Car', 'Bus', 'Truck'
car_df = df[df['tag'] == 'Car']
bus_df = df[df['tag'] == 'Bus']
truck_df = df[df['tag'] == 'Truck']

# change the type of 'cx', 'cy', 'cz', 'sx', 'sy', 'sz', 'yaw', 'mesh_cnt' into float
car_df['cx'] = car_df['cx'].astype(float)
car_df['cy'] = car_df['cy'].astype(float)
car_df['cz'] = car_df['cz'].astype(float)
car_df['sx'] = car_df['sx'].astype(float)
car_df['sy'] = car_df['sy'].astype(float)
car_df['sz'] = car_df['sz'].astype(float)
car_df['yaw'] = car_df['yaw'].astype(float)
car_df['mesh_cnt'] = car_df['mesh_cnt'].astype(float)
car_df['dist'] = car_df['dist'].astype(float)

bus_df['cx'] = bus_df['cx'].astype(float)
bus_df['cy'] = bus_df['cy'].astype(float)
bus_df['cz'] = bus_df['cz'].astype(float)
bus_df['sx'] = bus_df['sx'].astype(float)
bus_df['sy'] = bus_df['sy'].astype(float)
bus_df['sz'] = bus_df['sz'].astype(float)
bus_df['yaw'] = bus_df['yaw'].astype(float)
bus_df['mesh_cnt'] = bus_df['mesh_cnt'].astype(float)
bus_df['dist'] = bus_df['dist'].astype(float)

truck_df['cx'] = truck_df['cx'].astype(float)
truck_df['cy'] = truck_df['cy'].astype(float)
truck_df['cz'] = truck_df['cz'].astype(float)
truck_df['sx'] = truck_df['sx'].astype(float)
truck_df['sy'] = truck_df['sy'].astype(float)
truck_df['sz'] = truck_df['sz'].astype(float)
truck_df['yaw'] = truck_df['yaw'].astype(float)
truck_df['mesh_cnt'] = truck_df['mesh_cnt'].astype(float)
truck_df['dist'] = truck_df['dist'].astype(float)


car_df_filtered = car_df[(car_df['dist'] > 0) & (car_df['dist'] <= 60)]
truck_df_filtered = truck_df[(truck_df['dist'] > 0) & (truck_df['dist'] <= 60)]
bus_df_filtered = bus_df[(bus_df['dist'] > 0) & (bus_df['dist'] <= 60)]

x = car_df_filtered['dist'] / 60
yc = 1*(car_df_filtered['dist']**2 * (car_df_filtered['mesh_cnt']) 
    /np.sqrt(car_df_filtered['sx'] * car_df_filtered['sy'] * car_df_filtered['sz'])) /(1e6*(x * np.log(x) + 0.5))
x = truck_df_filtered['dist'] / 60
yt = 0.7*(truck_df_filtered['dist']**2 * (truck_df_filtered['mesh_cnt'])
    /np.sqrt(truck_df_filtered['sx'] * truck_df_filtered['sy'] * truck_df_filtered['sz'])) /(1e6*(x * np.log(x) + 0.5))
x = bus_df_filtered['dist'] / 60
yb = 0.9*(bus_df_filtered['dist']**2 * (bus_df_filtered['mesh_cnt'])
    /np.sqrt(bus_df_filtered['sx'] * bus_df_filtered['sy'] * bus_df_filtered['sz'])) /(1e6*(x * np.log(x) + 0.5))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(car_df_filtered['dist'],car_df_filtered['mesh_cnt'], yc, c='r', label='Car')
ax.scatter(truck_df_filtered['dist'],truck_df_filtered['mesh_cnt'], yt, c='b', label='Truck')
ax.scatter(bus_df_filtered['dist'],bus_df_filtered['mesh_cnt'], yb, c='g', label='Bus')
ax.set_xlabel('dist')
ax.set_ylabel('mesh_cnt')
ax.set_zlabel('s2 score')
ax.set_title('s2 score - mesh_cnt - mesh_cnt')
plt.show()