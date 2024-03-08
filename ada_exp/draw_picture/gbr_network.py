import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
set_label = ['50-25','75-37','100-50','125-67','150-75']
set_A_05 = ['1226_1700', '1226_1713', '1226_1727', '1226_1741', '1226_1756']
set_B_02 = ['0104_2223', '0104_2309', '0104_2328', '0104_2343', '0105_0013']
set_C_10 = ['0116_0117', '0116_0125', '0116_0132', '0116_0140', '0116_0149']
set_D_06 = ['0104_1949', '0104_2002', '0104_2016', '0104_2032', '0104_2056']


# B:8.04% -> 1.1
# D:9.15% -> 0.8

# df = pd.read_csv(s1s2_path)
# X = np.array([df['scan_enp'], df['bev_enp']]).T
# y = np.array(df['s1_gt'])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

def evaluate_regression(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, rmse, mae, r2





def z_standrand_scaler( y_pred, y_test):
    # draw the accuracy curve
    accuracys = []
    num_0_pred = []
    num_1_pred = []
    num_0_test = []
    num_1_test = []
    per = np.arange(0, 20, 0.4)
    for standard in per:
        y_pred2 = y_pred.copy()
        y_test2 = y_test.copy()
        y_pred2[y_pred2 <= standard] = 0
        y_pred2[y_pred2 > standard] = 1
        y_test2[y_test2 <= standard] = 0
        y_test2[y_test2 > standard] = 1
        accuracy = np.sum(y_pred2 == y_test2) / len(y_test2)
        accuracys.append(accuracy)
        num_0_pred.append(np.sum(y_pred2 == 0))
        num_1_pred.append(np.sum(y_pred2 == 1))
        num_0_test.append(np.sum(y_test2 == 0))
        num_1_test.append(np.sum(y_test2 == 1))

    # # draw the accuracy curve
    # plt.figure(figsize=(15,10))
    # plt.subplot(2,1,1)
    # plt.plot(per, accuracys)
    # plt.title('accuracy')
    # plt.xlabel('binarization standard')
    # plt.ylabel('predict accuracy')
    # for i in range(len(per)):
    #     plt.text(per[i], accuracys[i], str(round(accuracys[i], 3)))

    # plt.subplot(2,1,2)
    # # draw 0 and 1 in one figure
    # plt.plot(per, num_0_pred, label='num_0_pred')
    # plt.plot(per, num_1_pred, label='num_1_pred')
    # plt.plot(per, num_0_test, label='num_0_test')
    # plt.plot(per, num_1_test, label='num_1_test')
    # plt.xlabel('binarization standard')
    # plt.ylabel('sample number')
    # text the accuracy of each point
    # for i in range(len(per)):
    #     # plt.text(per[i], num_1_pred[i], str(round(accuracys[i], 3)))
    #     plt.text(per[i], num_0_pred[i], str(round(num_0_pred[i], 3)))
    #     plt.text(per[i], num_1_pred[i], str(round(num_1_pred[i], 3)))
    #     plt.text(per[i], num_0_test[i], str(round(num_0_test[i], 3)))
    #     plt.text(per[i], num_1_test[i], str(round(num_1_test[i], 3)))

    # plt.legend()
    # plt.title('num')
    # plt.show()

    return

def evaluate_regression(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, rmse, mae, r2


def test_on_local(filp,use,X_train, X_test, y_train, y_test):

    if use:
        index = np.where(y_train> filp)
        X_train = np.concatenate((X_train, X_train[index]), axis=0)
        y_train = np.concatenate((y_train, y_train[index]), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


    gbr = GradientBoostingRegressor(loss='huber',learning_rate=0.2)
    gbr.fit(X_train,y_train)
    y_pred = gbr.predict(X_train)
    gbr_mse, gbr_rmse, gbr_mae, gbr_r2 = evaluate_regression(y_train, y_pred)
    print('[GBR x and y] test on train dataset')
    print('mse: ', gbr_mse)
    print('rmse: ', gbr_rmse)
    print('mae: ', gbr_mae)
    print('r2: ', gbr_r2)

    y_pred = gbr.predict(X_test)
    gbr_mse, gbr_rmse, gbr_mae, gbr_r2 = evaluate_regression(y_test, y_pred)
    print('[GBR x and y] test on test dataset')
    print('mse: ', gbr_mse)
    print('rmse: ', gbr_rmse)
    print('mae: ', gbr_mae)
    print('r2: ', gbr_r2)

    z_standrand_scaler(y_pred, y_test)

    # vis the test data and curve in 3D

    #生成网格点
    X, Y = np.meshgrid(np.linspace(0.1, 1.6, 40), np.linspace(0, 0.12, 40))
    # X, Y = np.meshgrid(np.linspace(-1.1, 1.1, 30), np.linspace(-4.0, -1.9, 30))
    Z = gbr.predict(np.column_stack((X.ravel(), Y.ravel())))
    Z = Z.reshape(X.shape)
    # Z[Z <= 0] = np.nan
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(X_train[:,0], X_train[:,1], y_train, c='r', marker='o')
    ax.scatter(X_test[:,0], X_test[:,1], y_test, c='r', marker='o')
    ax.scatter(X_test[:,0], X_test[:,1], y_pred, c='b', marker='o')
 
    ax.set_xlabel('Scan Entropy')
    ax.set_ylabel('BEV Entropy')
    ax.set_zlabel('Predict Value')
    ax.set_title('S1 Scene Entropy Regression')
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.5)

    # elevation=20, azim=120
    ax.view_init(elev=13, azim=-40)
    plt.show()

subset = set_C_10


idx, scan_entropy_list, bev_entropy_list, pf_scalar_list, pf_vector_list, s1_gt =[], [], [], [], [], []
empty_file = []
sub_cnt = 0
for time_dirc in subset:
    sem_pt_path = '/home/newDisk/tool/carla_dataset_tool/raw_data/record_2024_'+time_dirc+'/vehicle.tesla.model3.master/velodyne_semantic/'
    file_list = os.listdir(sem_pt_path)
    file_list = [f for f in file_list if f.endswith('.txt')]
    file_list.sort()

    for file_name in file_list:
        gt = 0
        with open(sem_pt_path+file_name, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            if len(lines) ==1:
                empty_file.append(str(sub_cnt)+file_name[-9:-4])
            else:
                scores = last_line.split(' ')
                s1_gt.append(scores[0])
                scan_entropy_list.append(np.float64(scores[1]))
                bev_entropy_list.append(np.float64(scores[2]))
                pf_scalar_list.append(np.float64(scores[3]))
                pf_vector_list.append(np.float64(scores[4].replace('\n','')))
                idx.append(str(sub_cnt)+file_name[-9:-4])

    sub_cnt += 1

X = np.array([scan_entropy_list, bev_entropy_list],dtype=np.float64).T
y = np.array(s1_gt,dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
test_on_local(10,False,X_train, X_test, y_train, y_test)