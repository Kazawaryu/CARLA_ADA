import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing



def draw_3d_graph():
    source_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'
    end_path = '/vehicle.tesla.model3.master/filt.csv'
    csv_files = [source_dir + '0925_1508'+ end_path , source_dir + '0925_2105' + end_path ,
                 source_dir + '0925_1624' + end_path, source_dir + '0925_2039' + end_path ,
                 source_dir + '0925_1532' + end_path, source_dir + '0925_1643'+end_path ,
                 source_dir + '0925_1612' + end_path, source_dir + '0925_2024' + end_path ,
                 source_dir + '0925_1544' + end_path, source_dir + '0925_2053' + end_path ]
    cnts = ['300(case01)', '300(case02)', 
            '250(case01)', '250(case02)',
            '200(case01)', '200(case02)',
            '150(case01)', '150(case02)',
            '100(case01)', '100(case02)']


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0

    for path in csv_files:
        df = pd.read_csv(path)
        ax.scatter(df['x'], df['y'], df['z'], label=cnts[i])
        i+=1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()  
    plt.show()


def draw_3d_graph2():
    source_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'
    end_path = '/vehicle.tesla.model3.master/filt.csv'
    csv_files = [source_dir + '1011_2104'+ end_path ,
                source_dir + '1011_2129' + end_path,
                 source_dir + '1012_1456' + end_path,
                  source_dir+ '1012_1555' + end_path ]
    cnts = ['150-100', 
            '75-50',
            '100-60',
            '150-100']


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0

    for path in csv_files:
        df = pd.read_csv(path)
        ax.scatter(df['x'], df['y'], df['z'], label=cnts[i])
        i+=1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()  
    plt.show()

def combina_all_data_into_one_csv():
    combina_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/all_data.csv'
    # source_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'
    # end_path = '/vehicle.tesla.model3.master/filt.csv'
    # csv_files = [source_dir + '0925_1508'+ end_path , source_dir + '0925_2105' + end_path ,
    #              source_dir + '0925_1624' + end_path, source_dir + '0925_2039' + end_path ,
    #              source_dir + '0925_1532' + end_path, source_dir + '0925_1643'+end_path ,
    #              source_dir + '0925_1612' + end_path, source_dir + '0925_2024' + end_path ,
    #              source_dir + '0925_1544' + end_path, source_dir + '0925_2053' + end_path ]
    source_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'
    end_path = '/vehicle.tesla.model3.master/filt.csv'
    csv_files = [source_dir + '1011_2104'+ end_path ,
                source_dir + '1011_2129' + end_path,
                 source_dir + '1012_1456' + end_path,
                  source_dir+ '1012_1555' + end_path ]
    data_list = []


    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        data_list.append(data)

    all_data = pd.concat(data_list, axis=0)
    all_data.to_csv(combina_path, index=False)
    print("Data combined and saved to", combina_path)
    
    return

def show_test_curve():
    path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/all_data.csv'
    df = pd.read_csv(path)
    x = np.array(df['x'])
    y = np.array(df['y'])
    z = np.array(df['z'])
    gbr = GradientBoostingRegressor()
    X_train = np.column_stack((x, y))
    y_train = z

    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.2, random_state=0)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    gbr.fit(X_train, y_train)



    # # 创建示例数据
    # x = np.random.rand(100)
    # y = np.random.rand(100)
    # z = x**2 + y**2  # 示例拟合曲线

    # # 训练 GBR 模型
    # gbr = GradientBoostingRegressor()
    # gbr.fit(np.column_stack((x, y)), z)

    # 生成网格点
    X, Y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    Z = gbr.predict(np.column_stack((X.ravel(), Y.ravel())))
    Z = Z.reshape(X.shape)

    # 绘制3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', label='Data')  # 绘制原始数据点
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)  # 绘制拟合曲面
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('GBR Regression')
    ax.legend()
    plt.show()


def fit_func():
    path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/all_data.csv'
    df = pd.read_csv(path)
    X_train = np.array([df['x'], df['y']]).T
    y_train_o = np.array(df['z'])

    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train_o, test_size=0.2, random_state=0)


    gbr = GradientBoostingRegressor()
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    gbr.fit(X_train, y_train)
    X, Y = np.meshgrid(np.linspace(0, 3, 5), np.linspace(0, 0.14, 5))
    Z = gbr.predict(np.column_stack((X.ravel(), Y.ravel())))
    Z = Z.reshape(X.shape)

    # 绘制3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(df['x']), np.array(df['y']), np.array(df['z']), c='r', label='Data')  # 绘制原始数据点
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)  # 绘制拟合曲面
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('GBR Regression')
    ax.legend()
    plt.show()   


def make_func(path):
    # read data
    df = pd.read_csv(path)
    # draw 3d scatter graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x'], df['y'], df['z'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    # train model
    X = df[['x', 'y', 'x**2', 'y**2', 'x*y', 'x/y']]
    y = df['z']
    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))
    # output model parameters and r-squared
    print('Coefficients:', model.coef_)
    print('Intercept:', model.intercept_)
    print('R-squared:', r2)

if __name__ == "__main__":
    # combina_all_data_into_one_csv()
    # draw_3d_graph2()
    show_test_curve()