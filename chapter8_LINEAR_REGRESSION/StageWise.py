# @Time : 2019/7/23 10:51 
# @Author : HXT
# @File : StageWise.py 
# @Software: PyCharm
# ======================================================================================================================
# 向前逐步回归
import numpy as np
import chapter8_LINEAR_REGRESSION.Util as util


# 向前逐步回归(状态空间以贪心进行随机漫步)
# numiters:迭代次数,eps=系数调整步长
def stage_wise(numiters, Xmat, Ymat, eps=0.01):
    # 设置记录数组
    m, n = Xmat.shape
    thetaiters = np.zeros((numiters, n))

    # 设置相关容器
    theta = np.zeros((n, 1))
    curtheta = theta.copy()
    anstheta = np.zeros((n, 1))
    # 向前逐步回归
    for i in range(numiters):
        lowesterror = np.inf  # 无穷大
        print(theta.T)

        # 一轮比较
        for j in range(n):
            for sign in [-1, 1]:
                curtheta = theta.copy()  # 关键:注意数据的初始化
                curtheta[j] += sign * eps
                curYhyp = np.dot(Xmat, curtheta)
                curres = util.rss_error(Ymat, curYhyp)
                if curres < lowesterror:
                    lowesterror = curres
                    # print(lowesterror)
                    anstheta = curtheta.copy()

        theta = anstheta.copy()
        thetaiters[i, :] = theta.T

    return thetaiters


if __name__ == "__main__":
    Xdata, Ydata = util.load_dataset("abalone.txt")

    Xmat = np.array(Xdata)
    Xmean = np.mean(Xmat, axis=0)
    Xvar = np.var(Xmat, axis=0)
    Xmat = ((Xmat - Xmean) / Xvar)
    # Xmat = util.regularize(Xmat)

    Ymat = np.array(Ydata)
    Ymat = Ymat - np.mean(Ymat, axis=0)

    # Xmat = util.regularize(Xmat)
    # print(stage_wise(Xmat, Ymat, 0.01, 200))
    print("200 iters")
    stage_wise(200, Xmat, Ymat, 0.01)
    print("5000 iters")
    stage_wise(5000, Xmat, Ymat, 0.001)
