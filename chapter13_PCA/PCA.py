import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    fr = open(filename)
    stringarr = [line.strip().split("\t") for line in fr.readlines()]
    dataarr = [list(map(float, line)) for line in stringarr]
    return np.mat(dataarr)


# pca
# topNfeat 返回前topNfeat个特征
def pca(datamat, topNfeat=9999999):
    meanVals = np.mean(datamat, axis=0)  # 均值
    meanRemoved = datamat - meanVals  # 数据去除均值(基于当前数据集 维数: 1000,2)
    # 协方差矩阵
    covMat = np.cov(meanRemoved, rowvar=False)  # rowvar 表示meanRemoved中一行表示为一个样本
    eigvals, eigvects = np.linalg.eig(np.mat(covMat))  # eigvals特征值,eigvects特征向量

    # print("原矩阵", covMat, "\n----------\n", "特征向量", eigvects, "\n==\n", np.dot(covMat, eigvects[:, 1].reshape(2, 1)))
    # print("*****************************************")
    # print("特征值", eigvals, "\n----------\n", "特征向量", eigvects, "\n==\n", eigvals[1] * eigvects[:, 1].reshape(2, 1))

    eigvalind = np.argsort(eigvals)  # 对特征值(向量)排序(由小到大)返回索引
    eigvalind = eigvalind[:-(topNfeat + 1): -1]  # 逆序取数
    redeigvects = eigvects[:, eigvalind]  # 特征向量中取前eigvalind列

    subdatamat = meanRemoved * redeigvects  # 取数据中对应降维的几列(维数: m,k = m,n * n,k)
    recodatamat = (subdatamat * redeigvects.T) + meanVals  # 利用降维后的数据构建原始数据(维数: m,n = m,k * k,n + m,1)

    return subdatamat, recodatamat


#
if __name__ == "__main__":
    dataset = load_data("testSet.txt")
    print(dataset.shape)
    subdatamat, recodatamat = pca(dataset, 1)
    print(subdatamat.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataset[:, 0].flatten().A[0], dataset[:, 1].flatten().A[0], marker="^", s=90)
    ax.scatter(recodatamat[:, 0].flatten().A[0], recodatamat[:, 1].flatten().A[0], marker="o", s=50, c="red")
    plt.show()
