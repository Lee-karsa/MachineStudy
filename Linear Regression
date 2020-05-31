import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
the linear regression model for machine study
f(x) = theta_1 * x + theta_0
"""

def ReadingDataSets():
    Datainputstream = np.array(pd.read_csv(r"D:\桌面\data.csv")) # 读取文件
    DataX = Datainputstream[:, 0: -1].ravel() # 将数据传入到各自的维度中
    DataY = Datainputstream[:, -1]
    DataSetShape = Datainputstream.shape # 获取数据规模
    return DataX, DataY, DataSetShape

def average(sets): # 计算平均值
    aver = sum(sets) / np.array(sets).shape[0]
    return aver

def ParameterSolve(x,y,m):
    # 为了计算最小的欧氏距离，采用求偏导，计算出各个theta的最优解闭式
    theta_1, theta_0 = 0, 0#赋初值
    parameter_1, parameter_2, parameter_3, parameter_4 = 0, 0, 0, 0
    for i in range(m):
        parameter_1 += y[i] * (x[i] - average(x))
        parameter_2 += x[i]**2
        parameter_3 += x[i]
    theta_1 = parameter_1 / ( parameter_2 - (1/m) * (parameter_3 **2) ) # theta_1的闭式
    for i in range(m):
        parameter_4 += y[i] - theta_1 * x[i]
    theta_0 = (1/ m) * parameter_4#theta_0的闭式
    return theta_1, theta_0

def LossFormula(x,y,m,theta_1,theta_0):#计算损失函数的
    J = 0
    for i in range(m):
        h = theta_1 * x[i] + theta_0
        J += ( h - y[i] ) ** 2
    J /= (2 * m)
    return J

def PartialTheta(x,y,m,theta_1,theta_0):#计算偏导
    theta_1Partial = 0
    theta_0Partial = 0
    for i in range(m):
        theta_1Partial += (theta_1 * x[i] + theta_0 - y[i]) * x[i]
    theta_1Partial /= (1/m)
    for i in range(m):
        theta_0Partial += theta_1 * x[i] + theta_0 - y[i]
    theta_0Partial /= (1/m)
    return [theta_1Partial,theta_0Partial]

def GradientDescent(x,y,m,alpha = 0.01,theta_1 = 0,theta_0 = 0):
    MaxIteration = 1000#迭代次数
    counter = 0#计数器
    Mindiffer = 0.0000000000001#上一次损失值与本次损失值之差的最小阈值
    c = LossFormula(x,y,m,theta_1,theta_0)
    differ = c + 10#先赋初值
    theta_1sets = [theta_1]
    theta_0sets = [theta_0]
    Loss = [c]
    """
    当上一次损失值与本次损失值之差小于最小阈值，进行迭代
    每迭代一次，损失值都与上一次做差，以确定是否 过梯度
    求得梯度，在原来的基础上进行梯度下降
    """
    while (np.abs(differ - c) > Mindiffer and counter < MaxIteration):#当上一次损失值与本次损失值之差小于最小阈值，并且迭代
        differ = c

        upgradetheta_1 = alpha * PartialTheta(x,y,m,theta_1,theta_0)[0]#求得的一次theta的梯度值
        upgradetheta_0 = alpha * PartialTheta(x,y,m,theta_1,theta_0)[1]

        theta_1 -= upgradetheta_1
        theta_0 -= upgradetheta_0#在原来的基础上进行梯度下降

        theta_1sets.append(theta_1)
        theta_0sets.append(theta_0)
        Loss.append(LossFormula(x,y,m,theta_1,theta_0))
        c = LossFormula(x,y,m,theta_1,theta_0)
        counter += 1

    return {"theta_1":theta_1,"theta_1sets":theta_1sets,"theta_0":theta_0,"theta_0sets":theta_0sets,"losssets":Loss}

def DrawScatterandPredictionModel(x,y,theta_1,theta_0,newtheta):
    plt.figure("linear regression")
    plt.scatter(x, y)
    plt.plot(x,theta_1 * x + theta_0,lw=2,label="initital linear regression")
    plt.plot(x,newtheta["theta_1"] * x + newtheta["theta_0"],ls="--",lw=0.5,label="optimzed linear regression")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x,y,shape = ReadingDataSets()
    th1, th0 = ParameterSolve(x,y,shape[0])
    loss = GradientDescent(x,y,shape[0],alpha=0.01,theta_1=th1,theta_0=th0)
    print(loss)
    DrawScatterandPredictionModel(x,y,th1,th0,loss)
