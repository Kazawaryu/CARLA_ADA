import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import numpy
import math
import argparse


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(2, 10)       
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def read_parser():
    parser = argparse.ArgumentParser(description='filt a function')
    parser.add_argument('--source', '-s', type=str, help='source directory')
    args = parser.parse_args()
    source = args.source
    return source
    
if __name__ == '__main__':
    source = read_parser()
    # the secound col is the label, the third an forth col is the data
    train_data = []
    train_label = []

    with open(source, "r") as file:
        data = file.readlines()
    for i in range(1, len(data)):
        line = data[i].split(",")
        train_data.append([float(line[2]), float(line[3])])
        train_label.append(float(line[1]))

    train_data = torch.tensor(train_data)               
    train_label = torch.tensor(train_label)

    net = NET()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    for epoch in range(30000):
        output = net(train_data)
        loss = loss_func(output, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {}, loss = {}".format(epoch+1, loss.item()))

    # randomly select 20 data from the train data to test

    with torch.no_grad():
        # 生成n*2的测试输入数据  转换为张量
        test_data = torch.tensor(np.meshgrid(torch.linspace(-1, 1, 100), np.linspace(-1, 1, 100))).T.reshape(-1, 2)
        test_label = net(test_data.float())
        test_label = test_label.reshape(100, 100).numpy()


    fig = plt.figure()                 #初始化画板
    ax = plt.axes(projection = "3d")   #设定为3d模式
    ax.scatter(train_data[:, 0], train_data[:, 1], train_label)   #绘制散点图
    ax.set_xlabel = ("X1")
    ax.set_ylabel = ("X2")
    ax.set_zlable = ("Y")

    fig = plt.figure()
    bx = plt.axes(projection = "3d")
    bx.plot_surface(np.meshgrid(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100))[0],
                    np.meshgrid(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100))[1],
                    test_label)                         #绘制表面图
    bx.set_xlabel = ("X1")
    bx.set_ylabel = ("X2")
    bx.set_zlable = ("Y")

    plt.show()