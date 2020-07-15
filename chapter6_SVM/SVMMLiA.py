# @Time : 2019/8/6 15:48 
# @Author : HXT
# @File : SVMMLiA.py 
# @Software: PyCharm
# ======================================================================================================================
# 简单SMO实现SVM
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename):
    datamat, labelmat = [], []
    fr = open(filename)
    m = 0
    for line in fr.readlines():
        linearr = line.strip().split("\t")
        datamat.append([float(linearr[0]), float(linearr[1])])
        labelmat.append(float(linearr[2]))
        m += 1
    return datamat, np.array(labelmat).reshape(m, 1)


# 随机选择一个与i不同的支持向量
def select_j_rand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


# 根据取值范围调整约束量值
def clip_alpha(aj, H, L):
    if H < aj:  # 超过上限
        aj = H
    if aj < L:  # 超过下限
        aj = L
    return aj


# 简化SMO算法
# 求解 约束值, b
def smo_simple(datamatin, classlabels, C, toler, maxiter):  # 数据集,类别标签,常数C,容错率,收敛循环速度
    b = 0
    m, n = datamatin.shape
    alphas = np.zeros((m, 1))  # 约束值数组
    iter = 0
    while (iter < maxiter):
        alphapairschanged = 0
        for i in range(m):  # 遍历所有样本向量
            # 预测类别fxi
            fxi = np.float(np.dot((alphas * classlabels).T, np.dot(datamatin, datamatin[i, :].reshape(n, 1)))) + b
            ei = fxi - np.float(classlabels[i])  # 误差(供求导使用)
            if ((classlabels[i] * ei < -toler) and (alphas[i] < C)) or \
                    ((classlabels[i] * ei > toler) and (alphas[i] > 0)):  # 在一定容错率中允许更新
                j = select_j_rand(i, m)
                fxj = np.float(np.dot((alphas * classlabels).T, np.dot(datamatin, datamatin[j, :].reshape(n, 1)))) + b
                ej = fxj - np.float(classlabels[j])  # 误差(供求导使用)

                # 保存老版本便于同时更新
                alphaiold = alphas[i].copy()
                alphajold = alphas[j].copy()

                # 计算alpha2的取值范围
                if (classlabels[i] != classlabels[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if (L == H):
                    print("L==H")
                    continue  # 跳出内循环

                # 求导计算的中间量(表示核函数)
                eta = 2.0 * np.dot(datamatin[i, :].reshape(1, n), datamatin[j, :].reshape(n, 1)) - \
                      np.dot(datamatin[i, :].reshape(1, n), datamatin[i, :].reshape(n, 1)) - \
                      np.dot(datamatin[j, :].reshape(1, n), datamatin[j, :].reshape(n, 1))
                if eta[:] >= 0:
                    print("eta>=0")
                    continue

                # 更新约束值参数
                alphas[j] -= (classlabels[j] * (ei - ej) / eta.flatten()).flatten()
                # alphas[j] -= classlabels[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alphajold) < 0.00001:
                    print("alpha j 变化小")
                    continue
                alphas[i] += classlabels[j] * classlabels[i] * (alphajold - alphas[j])  # 求导后更新
                # 二分类问题求解偏移量b
                b1 = b - ei - np.dot(np.dot(classlabels[i], (alphas[i] - alphaiold)),
                                     np.dot(datamatin[i, :].reshape(1, n), datamatin[i, :].reshape(n, 1))) - \
                     np.dot(np.dot(classlabels[j], (alphas[j] - alphajold)),
                            np.dot(datamatin[i, :].reshape(1, n), datamatin[j, :].reshape(n, 1)))
                b2 = b - ej - np.dot(np.dot(classlabels[i], (alphas[i] - alphaiold)),
                                     np.dot(datamatin[i, :].reshape(1, n), datamatin[j, :].reshape(n, 1))) - \
                     np.dot(np.dot(classlabels[j], (alphas[j] - alphajold)),
                            np.dot(datamatin[j, :].reshape(1, n), datamatin[j, :].reshape(n, 1)))
                # 可以只用一个支持向量对应的值更新,也可以用平均值
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                # 记录这个样本是否更新
                alphapairschanged += 1
                print(" %d次迭代: %dth支持向量  改变%d次" % (iter, i, alphapairschanged))

        if alphapairschanged == 0:  # 收敛一次
            iter += 1
        else:
            iter = 0
        print("迭代次数: %d" % (iter))
    return b, alphas


# 获得拟合直线的参数
def get_theta(datamatin, classlabels, alphas):
    # 布尔标记是无维数组
    supportdata = datamin[alphas[:, 0] > 0, :]
    supportclass = classlabels[alphas[:, 0] > 0, :]
    supportalpha = alphas[alphas[:, 0] > 0, :]
    element = (supportdata * supportclass).T
    theta = np.dot(element, supportalpha)
    print(theta)
    return theta


# 画图
def showClassifer(dataMat, w, b):
    # 正样本
    data_plus = []
    # 负样本
    data_minus = []
    for i in range(len(dataMat)):
        if labelmat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
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
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
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
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')
    plt.show()


if __name__ == "__main__":
    data, labelmat = load_dataset("testSet.txt")
    datamin = np.array(data)
    # 训练
    b, alphas = smo_simple(datamin, labelmat, 0.6, 0.001, 40)
    # 输出参数
    print("b", b, "alphas>0", alphas[alphas > 0])
    # 输出支持向量
    print("支持向量:")
    for i in range(datamin.shape[0]):
        if alphas[i] > 0.0:
            print(datamin[i], labelmat[i])
    # theta参数
    theta = get_theta(datamin, labelmat, alphas)
    # 画图
    showClassifer(data,theta,b)