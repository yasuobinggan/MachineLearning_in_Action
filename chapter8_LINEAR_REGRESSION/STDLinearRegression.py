# @Time : 2019/7/7 8:49 
# @Author : HXT
# @File : STDLinearRegression.py
# @Software: PyCharm
# ======================================================================================================================
"""标准线性回归"""

import numpy as np
import matplotlib.pyplot as plt
import chapter8_LINEAR_REGRESSION.Util as util


# 函数会改变矩阵的值


# 线性回归使用的梯度下降
def gradient_descent(cost, Xmat, Ymat, theta, alpha):  # 梯度下降
    m = Xmat.shape[0]
    # while cost > 1e-4:  # 减去梯度
    #     theta = theta - alpha * 1 / m * Xmat.T * (Xmat * theta - Ymat)
    #     cost = float((Ymat - Xmat * theta).T * (Ymat - Xmat * theta))
    i = 0
    while i < 1000:
        theta -= alpha * 1 / m * Xmat.T * (Xmat * theta - Ymat)
        i += 1
    return theta


# 标准线性回归(矩阵操作X,Y大写)
# => theta
def std_linear_regression(Xarr, Yarr, option):  # 选择获得最优参数的方式 0,1
    Xmat = np.mat(Xarr)
    Ymat = np.mat(Yarr)

    # 参数全为0时的代价函数值
    theta = np.zeros((Xmat.shape[1], 1))
    cost = float((Ymat - Xmat * theta).T * (Ymat - Xmat * theta))  # aligned 矩阵维数相对应
    print("initial cost J(theta)", cost)

    # 方法0: 直接对代价函数求导(正定矩阵)
    if option == 0:
        theta = np.linalg.pinv(Xmat.T * Xmat) * Xmat.T * Ymat  # 列向量
        return theta
    # 方法1: 梯度下降(对数据进行预处理)(学习率)
    else:
        # 当前数据已经预处理过了
        # Xmat = np.hstack((np.ones((Xmat.shape[0], 1)), Xmat))
        # theta = np.vstack((theta, np.zeros((1, 1))))
        alpha = 0.1
        theta = gradient_descent(cost, Xmat, Ymat, theta, alpha)

        return theta


if __name__ == "__main__":
    Xdata, Ydata = util.load_dataset("ex0.txt")
    Xmat = np.mat(Xdata)
    Ymat = np.mat(Ydata)
    # 对两种方法进行测试
    theta = std_linear_regression(Xdata, Ydata, 0)
    print("std trained theta op0\n", theta)
    theta = std_linear_regression(Xdata, Ydata, 1)
    print("std theta op1\n", theta)
    # 预测数据
    # Yhyp = Xmat * theta
    # 画图

    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(Xmat[:, 1].flatten().A[0].shape, Ymat.T.flatten().A[0].shape)
    ax.scatter(Xmat[:, 1].flatten().A[0], Ymat.T.flatten().A[0])
    # 事先进行排序
    Xcopy = Xmat.copy()
    Xcopy.sort(0)
    Yhyp = Xcopy * theta
    ax.plot(Xcopy[:, 1], Yhyp, c="red")
    plt.show()
