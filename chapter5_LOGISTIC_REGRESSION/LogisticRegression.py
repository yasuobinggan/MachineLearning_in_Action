# @Time : 2019/7/24 11:05 
# @Author : HXT
# @File : LogisticRegression.py
# @Software: PyCharm
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def load_data(filename):
    fr = open(filename)
    datamat = []
    labelmat = []
    for line in fr.readlines():
        linelist = line.strip().split()  # 默认字符串形式 split默认参数以空格分割字符串
        curdata = [1.0]
        for i in range(len(linelist) - 1):
            ele = float(linelist[i])
            curdata.append(ele)
        datamat.append(curdata)

        labelmat.append(int(linelist[len(linelist) - 1]))

    datamat = np.array(datamat)
    labelmat = np.array(labelmat)
    labelmat = labelmat.reshape(labelmat.size, 1)

    # print(datamat, labelmat)
    return datamat, labelmat


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 利用梯度下降/上升求回归系数
def grad_desecnt(Xmat, Ymat, alpha=0.1, op=0):
    numiters = 500
    theta = np.ones((Xmat.shape[1], 1))
    for i in range(numiters):
        # 观察梯度上升与梯度下降
        theta -= alpha * (np.dot(Xmat.T, sigmoid(np.dot(Xmat, theta)) - Ymat) + op * theta)
        # theta += alpha * np.dot(Xmat.T, Ymat - sigmoid(np.dot(Xmat, theta)))

    return theta


# 反向传播求梯度系数
# def grad_descent():
#     pass

# 随机梯度下降
def stocgrad_descent(Xmat, Ymat, alpha=0.01):
    m, n = Xmat.shape
    theta = np.ones((n, 1))
    for index in range(200):
        for i in range(m):
            Yhyp = sigmoid(np.dot(Xmat[i], theta))
            cost = Yhyp.astype(np.float) - Ymat[i].astype(np.float)
            theta -= alpha * Xmat[i].reshape(Xmat[i].size, 1) * cost
    return theta


# 改进随机梯度下降(基于随机批处理)
def stocgrad_descent_ev(Xmat, Ymat, numiters=500):
    curXmat = Xmat.copy()
    m, n = curXmat.shape
    theta = np.ones((n, 1))
    for index in range(numiters):#
        for i in range(m):
            alpha = 4 / (1.0 + index + i) + 0.01
            randindex = int(np.random.uniform(0,m))# 随机选取一个样本
            Yhyp = sigmoid(np.dot(curXmat[randindex], theta))
            cost = Yhyp.astype(np.float) - Ymat[randindex].astype(np.float)# 相减的两个数组保证元素类型相同
            theta -= alpha * curXmat[randindex].reshape(curXmat[randindex].size, 1) * cost
            np.delete(curXmat,randindex,axis=0)# 删除这个样本
    return theta

# 输出拟合直线
def plot_bestfit(Xmat, Ymat, theta):
    # 数据个数
    m = Xmat.shape[0]
    # 正样本
    xcord1, ycord1 = [], []
    # 负样本
    xcord2, ycord2 = [], []
    # 根据数据集标签进行分类(用于画散点图)
    for i in range(m):
        if int(Ymat[i]) == 1:  # 1为正样本
            xcord1.append(Xmat[i, 1])
            ycord1.append(Xmat[i, 2])
        else:  # 0为负样本
            xcord2.append(Xmat[i, 1])
            ycord2.append(Xmat[i, 2])
    # 新建图框
    fig = plt.figure()
    # 添加subplot
    ax = fig.add_subplot(111)
    # 绘制正样本
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    # 绘制负样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
    # x轴坐标
    x = np.arange(-3.0, 3.0, 0.1)
    # w0*x0 + w1*x1 * w2*x2 = 0
    # x0 = 1, x1 = x, x2 = y
    y = (-theta[0] - theta[1] * x) / theta[2]

    ax.plot(x, y)

    plt.title('BestFit')
    plt.xlabel('x1')
    plt.ylabel('y2')
    # 显示
    plt.show()


if __name__ == "__main__":
    datamat, labelmat = load_data("testSet.txt")
    theta1 = grad_desecnt(datamat, labelmat, alpha=0.001, op=0)  # op 正则化参数
    print(theta1)
    plot_bestfit(datamat, labelmat, theta1)

    theta2 = stocgrad_descent_ev(datamat, labelmat, numiters=500)
    print(theta2)
    plot_bestfit(datamat, labelmat, theta2)
