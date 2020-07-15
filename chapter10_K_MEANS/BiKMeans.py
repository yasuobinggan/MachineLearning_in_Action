# 二分K均值分类
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    datamat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split("\t")
        fltline = list(map(float, curline))
        datamat.append(fltline)
    return np.mat(datamat)


# 计算距离的函数:veca,vecb向量
# 当前选择欧式距离
def distance(veca, vecb):
    return np.sqrt(np.sum(np.power(veca - vecb, 2)))


# 随机选择k个质心
def rand_cent(dataset, k):
    m, n = dataset.shape
    if m == 0:
        return []
    centroids = np.mat(np.zeros((k, n)))  # 聚类容器:k,n
    for j in range(n):  # 按特征方向为质心赋值
        minj = np.min(dataset[:, j])  # 选取当前列最小的
        rangej = np.float(np.max(dataset[:, j]) - minj)  # 当前一列的范围
        centroids[:, j] = minj + rangej * np.random.rand(k, 1)  # 随机选择,填充一列
    return centroids


# kmeans聚类
def kmeans(dataset, k, createcent=rand_cent, dist=distance):
    m, n = dataset.shape
    clusterassment = np.mat(np.zeros((m, 2)))  # 簇分配结果:第一列簇索引,第二列误差
    centroids = createcent(dataset, k)  # 随机初始化质心
    if centroids == []:
        return [], []
    clusterchanged = True  # 簇分配结果是否改变
    while clusterchanged:
        clusterchanged = False
        # -----------------------选择每个数据点最近的质心-----------------------
        for i in range(m):  # 选择每个数据点最近的质心
            mindist = np.inf
            minindex = -1
            for j in range(k):  # 遍历每个质心
                distji = dist(centroids[j, :], dataset[i, :])
                if distji < mindist:  # 获取最短距离
                    mindist = distji
                    minindex = j
            if clusterassment[i, 0] != minindex:  # 确认有数据点的质心更改
                clusterchanged = True
            clusterassment[i, :] = minindex, mindist ** 2  # 存储分配结果
        # -----------------------更新质心-----------------------
        # print(centroids)
        for cent in range(k):  # 遍历所有质心
            ptsinclust = dataset[np.nonzero(clusterassment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsinclust, axis=0)  # 更新质心坐标(几维特征是几维坐标)

    return centroids, clusterassment  # 簇中心,数据点的的簇分配结果


# 二分k均值聚类(bikmeans聚类)
def bi_kmeans(dataset, k, dist=distance):
    m, n = dataset.shape
    clusterassment = np.mat(np.zeros((m, 2)))  # 簇分类结果
    # 利用第一个均值创建一个簇中心
    centroid0 = np.mean(dataset, axis=0).tolist()[0]  # 对列操作
    # centroid0 = np.mean(dataset, axis=0)[:, 0]

    centlist = [centroid0]  # 簇列表
    # 初始化簇分配结果矩阵(以初始化簇中心)
    for i in range(m):
        clusterassment[i, 1] = distance(np.mat(centroid0), dataset[i, :]) ** 2

    while (len(centlist) < k):  # 簇的数目小于k时
        bestcentTosplit, bestnewcents, bestclusterassment = None, None, None
        minSSE = np.inf

        for i in range(len(centlist)):  # 尝试划分每一个当前已经存在的簇
            ptsincurrcluster = dataset[np.nonzero(clusterassment[:, 0].A == i)[0], :]  # 取当前簇中的数据
            Curcentroids, Splitclusterassment = kmeans(dataset=ptsincurrcluster, k=2, dist=dist)  # 对当前数据2均值聚类
            if Curcentroids == [] and Splitclusterassment == []:
                return np.array(centlist), clusterassment
            SSEsplit = np.sum(Splitclusterassment[:, 1])  # 以当前簇进行聚类部分的误差
            SSEnosplit = np.sum(clusterassment[np.nonzero(clusterassment[:, 0].A != i)[0], 1])  # 未以当前簇聚类部分的误差
            # print(SSEsplit, SSEnosplit)
            # 选取最佳质心点,用于while中的真正划分
            if (SSEsplit + SSEnosplit) < minSSE:  # 判断当前总误差是否是最小误差
                bestcentTosplit = i  # 第i类作为本类划分类
                bestnewcents = Curcentroids  # 划分后的簇质心(两个)
                bestclusterassment = Splitclusterassment.copy()  # 划分聚类后的簇分配结果
                minSSE = SSEsplit + SSEnosplit

        # 以下两条语句为簇质心编号使用硬编码
        # 将划分数据中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号连续不出现空缺
        bestclusterassment[np.nonzero(bestclusterassment[:, 0].A == 0)[0], 0] = bestcentTosplit
        # 数组过滤选出本次2-means聚类划分后类编号为1数据点，将这些数据点类编号变为当前类个数+1，作为新的一个聚类
        bestclusterassment[np.nonzero(bestclusterassment[:, 0].A == 1)[0], 0] = len(centlist)

        # print("当前二分的 簇分类编号是", bestcentTosplit, "当前簇分类的数据数量", len(bestclusterassment))  # 输出

        # 更新质心列表
        centlist[bestcentTosplit] = bestnewcents[0, :]  # 更新质心列表中变化后的质心向量(编号0)
        centlist.append(bestnewcents[1, :])  # 加入新类的质心向量(编号1)
        # 更新簇分配结果(clusterAssment列表)中参与2-means聚类数据点变化后的分类编号，及数据距离该类的误差平方
        clusterassment[np.nonzero(clusterassment[:, 0].A == bestcentTosplit)[0], :] = bestclusterassment

    return np.array(centlist), clusterassment


def plot_dataset(datMat, myCentroids, clustAssing):
    clustAssing = clustAssing.tolist()
    myCentroids = myCentroids.tolist()
    xcord = [[], [], [], []]
    ycord = [[], [], [], []]
    datMat = datMat.tolist()
    m = len(clustAssing)
    for i in range(m):
        if int(clustAssing[i][0]) == 0:
            xcord[0].append(datMat[i][0])
            ycord[0].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 1:
            xcord[1].append(datMat[i][0])
            ycord[1].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 2:
            xcord[2].append(datMat[i][0])
            ycord[2].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 3:
            xcord[3].append(datMat[i][0])
            ycord[3].append(datMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制样本点
    ax.scatter(xcord[0], ycord[0], s=20, c='b', marker='*', alpha=.5)
    ax.scatter(xcord[1], ycord[1], s=20, c='r', marker='D', alpha=.5)
    ax.scatter(xcord[2], ycord[2], s=20, c='c', marker='>', alpha=.5)
    ax.scatter(xcord[3], ycord[3], s=20, c='k', marker='o', alpha=.5)
    # 绘制质心
    ax.scatter(myCentroids[0][0], myCentroids[0][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[1][0], myCentroids[1][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[2][0], myCentroids[2][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[3][0], myCentroids[3][1], s=100, c='k', marker='+', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


if __name__ == "__main__":
    dataset = load_data("testSet.txt")
    mycentroids, clustassing = bi_kmeans(dataset, 4)
    print("mycentroids\n", mycentroids)
    print("clustassing\n", clustassing)

    # 画图
    mycentroids = np.mat(mycentroids)  # 矩阵化
    plot_dataset(dataset, mycentroids, clustassing)
