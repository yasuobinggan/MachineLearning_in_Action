# kmeans
# mat矩阵分片切割后是有维数组
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
    centroids = np.mat(np.zeros((k, n)))  # 聚类容器:k,n
    for j in range(n):  # 按特征方向为质心赋值
        # print(dataset[:, j].shape)
        minj = np.min(dataset[:, j])  # 选取当前列最小的
        rangej = np.float(np.max(dataset[:, j]) - minj)  # 当前一列的范围
        centroids[:, j] = minj + rangej * np.random.rand(k, 1)  # 随机选择,填充一列
    return centroids


# kmeans聚类
def kmeans(dataset, k, createcent=rand_cent, dist=distance):
    m, n = dataset.shape
    clusterassment = np.mat(np.zeros((m, 2)))  # 簇分配结果:第一列簇索引,第二列误差
    centroids = createcent(dataset, k)  # 随机初始化质心
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
            centroids[cent, :] = np.mean(ptsinclust, axis=0)

    return centroids, clusterassment


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
    # print(dataset)
    # rand_cent(dataset, 2)
    # print(min(dataset[:, 0]))
    # print(max(dataset[:, 0]))
    # print(min(dataset[:, 1]))
    # print(max(dataset[:, 1]))
    #
    # print(rand_cent(dataset, 2))
    # print(distance(dataset[0], dataset[1]))
    mycentroids, clustassing = kmeans(dataset, 4)
    print(clustassing)
    # print(mycentroids)

    plot_dataset(dataset, mycentroids, clustassing)
