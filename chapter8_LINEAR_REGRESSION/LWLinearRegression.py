# @Time : 2019/7/14 11:05 
# @Author : HXT
# @File : LWLinearRegression.py 
# @Software: PyCharm
# ======================================================================================================================
"""局部加权线性回归"""

import numpy as np
import matplotlib.pyplot as plt
import chapter8_LINEAR_REGRESSION.Util as util  # 导入数据


# 局部加权线性回归
def lw_linear_regression(testpoint, Xdata, Ydata, k=1.0):
    Xmat = np.mat(Xdata)
    Ymat = np.mat(Ydata)

    m = Xmat.shape[0]
    # 创建加权对角阵
    weights = np.mat(np.eye(m))  # 单位阵
    # 求解权重矩阵(循环)
    for i in range(m):
        Diffmat = testpoint - Xmat[i, :]  # 求向量差量
        weights[i, i] = np.exp((Diffmat * Diffmat.T) / (-2 * k * k))  # 高斯核

    # 计算求导后结果(不明白)
    # print(Xmat.shape)
    # print(weights.shape)
    # print(Ymat.shape)
    theta = np.linalg.pinv(Xmat.T * weights * Xmat) * Xmat.T * weights * Ymat
    # print(theta,"\n")
    return testpoint * theta


# 对每一个点进行测试
def lwlrtest(testdata, Xdata, Ydata, k=1.0):
    m = np.mat(testdata).shape[0]
    Yhyp = np.zeros(m)  # 一维数组
    for i in range(m):
        Yhyp[i] = lw_linear_regression(testdata[i], Xdata, Ydata, k)
    return Yhyp


# 求解向量化(WRONG)
def MY_lwlrtest(testdata, Xdata, Ydata, k=1.0):
    testpoint = np.mat(testdata)
    Xmat = np.mat(Xdata)
    Ymat = np.mat(Ydata)
    m = Xmat.shape[0]
    # 创建加权对角阵
    weights = np.mat(np.eye(m))  # 单位阵
    # 求解权重矩阵(循环)
    Diffmat = testpoint - Xmat  # 求向量差量
    weights = np.exp((Diffmat * Diffmat.T) / (-2 * k * k))  # 高斯核

    # 计算求导后结果(不明白)
    theta = (Xmat.T * (weights * Xmat)).I * (Xmat.T * (weights * Ymat))
    # theta = np.linalg.pinv(Xmat.T * weights * Xmat) * Xmat.T * weights * Ymat
    Yhyp = testpoint * theta
    return Yhyp


if __name__ == "__main__":
    Xdata, Ydata = util.load_dataset("ex0.txt")

    # 基于不同的权重测试
    Yhyp1 = lwlrtest(Xdata, Xdata, Ydata, 1.0)
    Yhyp2 = lwlrtest(Xdata, Xdata, Ydata, 0.01)
    Yhyp3 = lwlrtest(Xdata, Xdata, Ydata, 0.003)

    """画图"""
    # 用于画图
    Xmat = np.mat(Xdata)
    Ymat = np.mat(Ydata)

    # 排序，返回索引值
    srtInd = Xmat[:, 1].argsort(0)
    xSort = Xmat[srtInd][:, 0, :]
    # 画图
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], Yhyp1[srtInd], c='b')
    axs[1].plot(xSort[:, 1], Yhyp2[srtInd], c='b')
    axs[2].plot(xSort[:, 1], Yhyp3[srtInd], c='b')
    axs[0].scatter(Xmat[:, 1].flatten().A[0], Ymat.flatten().A[0], s=20, c='r', alpha=.5)
    axs[1].scatter(Xmat[:, 1].flatten().A[0], Ymat.flatten().A[0], s=20, c='r', alpha=.5)
    axs[2].scatter(Xmat[:, 1].flatten().A[0], Ymat.flatten().A[0], s=20, c='r', alpha=.5)
    plt.xlabel('X')
    plt.show()
