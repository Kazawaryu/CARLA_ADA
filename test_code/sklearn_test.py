import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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

def combina_all_data_into_one_csv():
    combina_path = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/all_data.csv'
    source_dir = '/home/ghosnp/project/fix_space/origin/carla_dataset_tools/raw_data/record_2023_'
    end_path = '/vehicle.tesla.model3.master/filt.csv'
    csv_files = [source_dir + '0925_1508'+ end_path , source_dir + '0925_2105' + end_path ,
                 source_dir + '0925_1624' + end_path, source_dir + '0925_2039' + end_path ,
                 source_dir + '0925_1532' + end_path, source_dir + '0925_1643'+end_path ,
                 source_dir + '0925_1612' + end_path, source_dir + '0925_2024' + end_path ,
                 source_dir + '0925_1544' + end_path, source_dir + '0925_2053' + end_path ]
    
    data_list = []


    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        data_list.append(data)

    all_data = pd.concat(data_list, axis=0)
    all_data.to_csv(combina_path, index=False)
    print("Data combined and saved to", combina_path)
    
    return


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
    combina_all_data_into_one_csv()