# TODO: read last dataset label count, entropy, draw grahps

import os
import shutil
import argparse
import matplotlib.pyplot as plt

def read_data(source_dir):
    # only read the last line of the file
    with open(source_dir+"/log.txt", "r") as file:
        data = file.readlines()[-1]
    return data

def read_parser():
    parser = argparse.ArgumentParser(description='Format Helper')
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
    

if __name__ == "__main__":
    source = read_parser()
    x_axis = []
    temp_vehicle_cnt = []
    scan_entropy = []
    bev_entropy = []
    current_entropy_score = []

    idx = 0
    for source_dir in source:
        data = read_data(source_dir).split(" ")
        x_axis.append(idx)
        temp_vehicle_cnt.append(int(data[0]))
        scan_entropy.append(float(data[1]))
        bev_entropy.append(float(data[2]))
        current_entropy_score.append(float(data[3]))
        idx += 1

    draw_graph(x_axis, temp_vehicle_cnt, "frame number", "vehicle count", "vehicle count")
    draw_graph(x_axis, scan_entropy, "frame number", "scan entropy", "scan entropy")
    draw_graph(x_axis, bev_entropy, "frame number", "bev entropy", "bev entropy")
    draw_graph(x_axis, current_entropy_score, "frame number", "current entropy score", "current entropy score")


        