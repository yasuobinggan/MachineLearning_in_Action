# @Time : 2019/8/9 19:57 
# @Author : HXT
# @File : KERNEL.py 
# @Software: PyCharm
# ======================================================================================================================
# 基于kernel的SVM
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


# 数据结构
class OptStruct():  # 对应变量OS为结构体
    def __init__(self, datamatin, classlabels, C, toler, KTup):  # KTup核参数
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
        self.Kernel = np.mat(np.zeros((self.m, self.m)))
        # 填充核函数矩阵 TODO
        for i in range(self.m):
            self.Kernel[:, i] = kernel(self.Xdata, self.Xdata[i, :], KTup)


# 核方法
def kernel(Xdata, sample, KTup):
    m, n = Xdata.shape
    K = np.mat(np.zeros((m, 1)))
    if KTup[0] == "lin":  # 线性核
        K = Xdata * sample.T
    elif KTup[0] == "rbf":  # 高斯核
        # 计算分子
        for i in range(m):
            deltarow = Xdata[i, :] - sample
            K[i] = deltarow * deltarow.T
        # 整体计算
        K = np.exp(-K / KTup[1] ** 2)
    else:
        raise NameError("Kernel is not recognized")

    return K


def calc_Ek(OS, k):
    fxk = np.float(np.multiply(OS.alphas, OS.Ylabel).T * OS.Kernel[:, k]) + OS.b
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
        eta = 2.0 * OS.Kernel[i, j] - OS.Kernel[i, i] - OS.Kernel[j, j]
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
        b1 = OS.b - Ei - OS.Ylabel[i] * (OS.alphas[i] - alphaiold) * OS.Kernel[i, i] - OS.Ylabel[j] * \
             (OS.alphas[j] - alphajold) * OS.Kernel[i, j]
        b2 = OS.b - Ej - OS.Ylabel[i] * (OS.alphas[i] - alphaiold) * OS.Kernel[i, j] - OS.Ylabel[j] * \
             (OS.alphas[j] - alphajold) * OS.Kernel[j, j]
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
def smo(datamatin, labellist, C, toler, maxiter, Ktup=("lin", 0)):
    OS = OptStruct(np.mat(datamatin), np.mat(labellist).T, C, toler, Ktup)
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


def testRBF(filename, alphas, b, SVind, SVs, labelSV, k1=1.3):
    datalist, labellist = load_dataset(filename)
    Xdata, Ylabel = np.mat(datalist), np.mat(labellist).T
    m, n = Xdata.shape
    errorcount = 0
    for i in range(m):
        kerneleval = kernel(SVs, Xdata[i, :], KTup=("rbf", k1))
        predict = kerneleval.T * np.multiply(labelSV, alphas[SVind]) + b
        if np.sign(predict) != np.sign(labellist[i]):
            errorcount += 1
    print(filename, "误差 %f" % (np.float(errorcount) / m))


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

    # 找出支持向量点
    # enumerate在字典上是枚举、列举的意思
    for i, alpha in enumerate(alphas):
        # 支持向量机的点
        if (abs(alpha) > 0):
            x, y = datalist[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')
    plt.show()


if __name__ == "__main__":
    datalist, labellist = load_dataset("testSetRBF.txt")
    alphas, b = smo(datalist, labellist, 200, 0.0001, 10000, ("rbf", 1.3))
    theta = get_theta(alphas, datalist, labellist)
    showClassifer(datalist, labellist, theta, b)

    Xdata, Ylabel = np.mat(datalist), np.mat(labellist).T
    SVind = np.nonzero(alphas.A > 0)[0]  # 支持向量的序号
    SVs = Xdata[SVind]  # 支持向量
    labelSV = Ylabel[SVind]  # 支持向量对应的标签
    print("支持向量的数量%d" % (SVs.shape[0]))
    testRBF("testSetRBF.txt", alphas, b, SVind, SVs, labelSV, k1=1.3)
    testRBF("testSetRBF2.txt", alphas, b, SVind, SVs, labelSV, k1=1.3)
