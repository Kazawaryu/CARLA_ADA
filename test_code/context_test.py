# TODO: read last dataset label count, entropy, draw grahps

import os
import shutil
import argparse
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def read_data(source_dir):
    # only read the last line of the file
    with open(source_dir, "r") as file:
        data = file.readlines()[-1]
    return data

def read_parser():
    parser = argparse.ArgumentParser(description='save desc info')
    parser.add_argument('--source', '-s', type=str, help='source directory')
    args = parser.parse_args()
    source = args.source
    return source

def draw_graph(x_axis, y_axis, x_label, y_label, title):
    plt.plot(x_axis, y_axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def make_linear_func(path):
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
    source = read_parser()
    x_axis = []
    temp_vehicle_cnt = []
    scan_entropy = []
    bev_entropy = []
    current_entropy_score = []

    vehicle_0_2 = []
    vehicle_2_4 = []
    vehicle_4_6 = []
    vehicle_6_8 = []
    vehicle_8 = []

    idx = 0
    for filename in os.listdir(source):
        if filename.endswith(".txt"):
            
            data = read_data(source+filename).split(" ")
            x_axis.append(idx)
            temp_vehicle_cnt.append(float(data[0]))
            scan_entropy.append(float(data[1]))
            bev_entropy.append(float(data[2]))
            current_entropy_score.append(float(data[3]))
            idx += 1

    # save data to csv file
    with open(source+"../filt.csv", "w") as file:
        file.write("x,y,z,x**2,y**2,x/y,x*y\n")
        for i in range(len(x_axis)):
            file.write("{},{},{},{},{},{},{}\n".format(scan_entropy[i], bev_entropy[i], temp_vehicle_cnt[i],scan_entropy[i]**2, bev_entropy[i]**2, current_entropy_score[i], scan_entropy[i]*bev_entropy[i]))

    make_linear_func(source+"../filt.csv")