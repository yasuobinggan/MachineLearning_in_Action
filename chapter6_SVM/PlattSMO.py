# @Time : 2019/8/8 21:31 
# @Author : HXT
# @File : PlattSMO.py
# @Software: PyCharm
# ======================================================================================================================
# 完整版SMO算法
# 使用numpy矩阵
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename):
    datalist, labellist = [], []
    fr = open(filename)
    m = 0
    for line in fr.readlines():
        linearr = line.strip().split("\t")
        datalist.append([float(linearr[0]), float(linearr[1])])
        labellist.append(float(linearr[2]))
        m += 1
    return datalist, labellist  # np.array(labelmat).reshape(m, 1)


###辅助函数###
# 表达支持向量机的数据结构
class OptStruct():  # 对应变量OS为结构体
    def __init__(self, datamatin, classlabels, C, toler):
        self.Xdata = datamatin
        self.Ylabel = classlabels
        self.C = C
        self.tol = toler
        self.m = datamatin.shape[0]
        self.n = datamatin.shape[1]
        # 参数
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.Ecache = np.mat(np.zeros((self.m, 2)))  # 第一列Ecache是否有效,第二列实际的损失值


# 计算第k个支持向量的误差
def calc_Ek(OS, k):
    fxk = np.float(np.multiply(OS.alphas, OS.Ylabel).T * (OS.Xdata * OS.Xdata[k, :].T)) + OS.b
    Ek = fxk - np.float(OS.Ylabel[k])
    return Ek


# 对OS.Ecache进行更新
def update_Ek(OS, k):
    Ek = calc_Ek(OS, k)
    OS.Ecache[k] = [1, Ek]


# 随机选择j
def select_j_rand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


# 基于最大间隔选择向量
def select_j(OS, i, Ei):
    maxj, maxdeltaE = -1, 0
    Ej = 0
    validEcachelist = np.nonzero(OS.Ecache[:, 0].A)[0]  # 返回一个误差非零的索引(非零索引计算误差间隔才有效)
    if len(validEcachelist) > 1:
        for k in validEcachelist:
            if i == k:
                continue
            Ek = calc_Ek(OS, k)
            if np.abs(Ek - Ei) > maxdeltaE:
                maxj = k
                Ej = Ek
                maxdeltaE = np.abs(Ek - Ei)
        return maxj, Ej
    else:
        j = select_j_rand(i, OS.m)
        Ej = calc_Ek(OS, j)
        return j, Ej


# 根据取值范围调整约束量值
def clip_alpha(aj, H, L):
    if H < aj:  # 超过上限
        aj = H
    if aj < L:  # 超过下限
        aj = L
    return aj


######

# 内循环
# 返回是否改变
def innerloop(OS, i):
    Ei = calc_Ek(OS, i)
    OS.Ecache[i] = [1, Ei]
    if (OS.Ylabel[i] * Ei < -OS.tol and OS.alphas[i] < OS.C) or (OS.Ylabel[i] * Ei > OS.tol and OS.alphas[i] > 0):
        j, Ej = select_j(OS, i, Ei)

        alphaiold = OS.alphas[i].copy()
        alphajold = OS.alphas[j].copy()
        # alphaj的取值范围
        if OS.Ylabel[i] != OS.Ylabel[j]:
            L = max(0, OS.alphas[j] - OS.alphas[i])
            H = min(OS.C, OS.C + OS.alphas[j] - OS.alphas[i])
        else:
            L = max(0, OS.alphas[j] + OS.alphas[i] - OS.C)
            H = min(OS.C, OS.alphas[j] + OS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * OS.Xdata[i, :] * OS.Xdata[j, :].T - OS.Xdata[i, :] * OS.Xdata[i, :].T \
              - OS.Xdata[j, :] * OS.Xdata[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        OS.alphas[j] -= OS.Ylabel[j] * (Ei - Ej) / eta
        OS.alphas[j] = clip_alpha(OS.alphas[j], H, L)
        update_Ek(OS, j)
        if np.abs(OS.alphas[j] - alphajold) < 0.00001:
            print("j改变过小")
            return 0
        OS.alphas[i] += OS.Ylabel[j] * OS.Ylabel[i] * (alphajold - OS.alphas[j])
        update_Ek(OS, i)

        # 二分类问题求解偏移量b
        b1 = OS.b - Ei - OS.Ylabel[i] * (OS.alphas[i] - alphaiold) * OS.Xdata[i, :] * OS.Xdata[i, :].T - OS.Ylabel[j] * \
             (OS.alphas[j] - alphajold) * OS.Xdata[i, :] * OS.Xdata[j, :].T
        b2 = OS.b - Ej - OS.Ylabel[i] * (OS.alphas[i] - alphaiold) * OS.Xdata[i, :] * OS.Xdata[j, :].T - OS.Ylabel[j] * \
             (OS.alphas[j] - alphajold) * OS.Xdata[j, :] * OS.Xdata[j, :].T
        # 可以只用一个支持向量对应的值更新,也可以用平均值
        if (0 < OS.alphas[i]) and (OS.C > OS.alphas[i]):
            OS.b = b1
        elif (0 < OS.alphas[j]) and (OS.C > OS.alphas[j]):
            OS.b = b2
        else:
            OS.b = (b1 + b2) / 2.0

        return 1
    else:
        return 0


# 外循环
def smo(datamatin, labellist, C, toler, maxiter, ktup=("lin", 0)):
    OS = OptStruct(np.mat(datamatin), np.mat(labellist).T, C, toler)
    ## 控制循环的三个变量
    iter = 0
    entireset = True
    alphapairschanged = 0
    while (iter < maxiter) and ((alphapairschanged > 0) or (entireset)):
        alphapairschanged = 0
        if entireset:  # 整个数据集
            for i in range(OS.m):
                alphapairschanged += innerloop(OS, i)
                print("FULLSET 第%d次迭代 %dth样本调整%d次" % (iter, i, alphapairschanged))
            iter += 1
        else:  # 非边界数据集
            nonboundis = np.nonzero((OS.alphas.A > 0) * (OS.alphas.A < C))[0]
            for i in nonboundis:
                alphapairschanged += innerloop(OS, i)
                print("NON-BOUND 第%d次迭代 %dth样本调整%d次" % (iter, i, alphapairschanged))
            iter += 1

        # 加速手段
        if entireset:  # 这一次整个数据集,下一次边界即可
            entireset = False
        elif alphapairschanged == 0:  # alpha未改变下一次仍对整个数据集进行改变
            entireset = True
        print("迭代次数 %d" % (iter))

    return OS.alphas, OS.b


# 获得参数
def get_theta(alphas, datalist, labellist):
    Xdata = np.mat(datalist)
    Ylabel = np.mat(labellist).T
    m, n = Xdata.shape
    theta = np.zeros((n, 1))
    for i in range(m):
        theta += np.multiply(alphas[i] * Ylabel[i], Xdata[i, :].T)
    return theta


# 画图
def showClassifer(datalist, classlist, theta, b):
    # 正样本
    data_plus = []
    # 负样本
    data_minus = []
    for i in range(len(datalist)):
        if classlist[i] > 0:
            data_plus.append(datalist[i])
        else:
            data_minus.append(datalist[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)
    # 正样本散点图（scatter）
    # transpose转置
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    # 负样本散点图（scatter）
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    # 绘制直线
    x1 = max(datalist)[0]
    x2 = min(datalist)[0]
    a1, a2 = theta
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    # enumerate在字典上是枚举、列举的意思
    for i, alpha in enumerate(alphas):
        # 支持向量机的点
        if (abs(alpha) > 0):
            x, y = datalist[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')
    plt.show()


if __name__ == "__main__":
    datalist, labellist = load_dataset("testSet.txt")
    alphas, b = smo(datalist, labellist, 0.6, 0.001, 40)
    theta = get_theta(alphas, datalist, labellist)
    showClassifer(datalist, labellist, theta, b)
