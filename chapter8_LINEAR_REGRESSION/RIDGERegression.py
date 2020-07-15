# @Time : 2019/7/23 9:38 
# @Author : HXT
# @File : RIDGERegression.py 
# @Software: PyCharm
# ======================================================================================================================
# 岭回归
import numpy as np
import matplotlib.pyplot as plt
import chapter8_LINEAR_REGRESSION.Util as util


# 梯度下降
def gradient_descent(Xmat, Ymat, alpha=np.exp(-10)):
    m = Xmat.shape[0]
    numiters = 10
    theta = np.ones((Xmat.shape[1], 1))
    for i in range(numiters):
        theta -= alpha * 1 / m * Xmat.T * (Xmat * theta - Ymat) + alpha * theta
    # print(theta)
    return theta


# 直接求导
def ridge_regression(lam, Xmat, Ymat):
    theta = np.linalg.pinv(Xmat.T * Xmat + lam * np.eye(Xmat.shape[1])) * Xmat.T * Ymat
    return theta


# 对30个不同的lambda进行测试
def ridgeregression_test(Xmat, Ymat):
    # 对30个不同的lambda进行测试
    numtestpts = 30
    THETA1 = np.zeros((numtestpts, Xmat.shape[1]))  # 基于不同lambda的theta矩阵
    THETA2 = np.zeros((numtestpts, Xmat.shape[1]))  # 基于不同lambda的theta矩阵

    for i in range(numtestpts):
        THETA1[i, :] = ridge_regression(np.exp(i - 10), Xmat, Ymat).T
    for i in range(numtestpts - 10):
        THETA2[i, :] = ridge_regression(np.exp(i - 10), Xmat, Ymat).T
    return THETA1, THETA2


if __name__ == "__main__":
    Xdata, Ydata = util.load_dataset("abalone.txt")
    Xmat = np.mat(Xdata)
    Ymat = np.mat(Ydata)
    # 数据预处理
    Ymean = np.mean(Ymat, axis=0)
    Ymat = Ymat - Ymean

    Xmean = np.mean(Xmat, axis=0)
    Xvar = np.var(Xmat, axis=0)
    Xmat = (Xmat - Xmean) / Xvar
    # 梯度下降测试
    # theta = gradient_descent(Xmat, Ymat)
    # print(theta)

    # 基于不同的惩罚程度测试 # 梯度下降并不是一个好方法
    THETA1, THETA2 = ridgeregression_test(Xmat, Ymat)
    print("直接求导\n", THETA1)
    print("梯度下降\n", THETA2)

    fig1 = plt.figure()
    plt.plot(THETA1)
    fig2 = plt.figure()
    plt.plot(THETA2)
    plt.show()

    # 画图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # print(Xmat[:, 1].flatten().A[0].shape, Ymat.T.flatten().A[0].shape)
    # ax.scatter(Xmat[:, 1].flatten().A[0], Ymat.T.flatten().A[0])
    # # 事先进行排序
    # Xcopy = Xmat.copy()
    # Xcopy.sort(0)
    # Yhyp = Xcopy * THETA.T[:,0]  # 预测值
    # ax.plot(Xcopy[:, 1], Yhyp, c="red")
    # plt.show()
