# @Time : 2019/8/13 16:08 
# @Author : HXT
# @File : KNN.py 
# @Software: PyCharm
# ======================================================================================================================
# KNN
import numpy as np
import matplotlib.pyplot as plt
import operator


def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


# 读入数据
def load_dataset(filename):
    fr = open(filename)
    datalist, labellist = [], []
    for line in fr.readlines():
        arrarylist = line.strip().split("\t")
        arrarylist = list(map(float, arrarylist))
        datalist.append(arrarylist[:3])
        labellist.append(int(arrarylist[-1]))
    return np.array(datalist), labellist


def load_dataset_1(filename):
    fr = open(filename)
    datalist, labellist = [], []
    for line in fr.readlines():
        arrarylist = line.strip().split("\t")
        subarrarylist = list(map(float, arrarylist[:3]))
        datalist.append(subarrarylist[:3])
        labellist.append(arrarylist[-1])
    return np.array(datalist), labellist


# 特征归一化
def auto_norm(dataset):
    minvals = np.min(dataset, axis=0)
    maxvals = np.max(dataset, axis=0)
    # print(minvals, maxvals)
    ranges = maxvals - minvals
    m, n = dataset.shape
    normdataset = np.zeros_like(dataset)
    normdataset = dataset - minvals
    normdataset = normdataset / ranges
    return normdataset, ranges, minvals


# knn分类器
def knn_classify(inX, Xdata, Ylabel, k):
    m = Xdata.shape[0]
    diffmat = (np.tile(inX, (m, 1)) - Xdata) ** 2  # 计算距离
    distances = np.sum(diffmat, axis=1)  # 返回一个无维数组
    sorteddistanceindicies = np.argsort(distances, axis=0)
    classcount = {}  # 一个类别对应一个投票位置
    for i in range(k):  # 对k个最近邻进行投票
        voteIlabel = Ylabel[sorteddistanceindicies[i]]  # 取出应该投票的类别
        classcount[voteIlabel] = classcount.get(voteIlabel, 0) + 1  # 投票
    # 选择最大投票的类
    maxlabel = None
    maxvote = -np.inf
    for key, value in classcount.items():
        if value > maxvote:
            maxlabel = key
            maxvote = value
    return maxlabel


# 测试knn
def datingclass_test():
    horatio = 0.10  # 测试集所用比例
    datingdatamat, datinglabels = load_dataset_1("datingTestSet.txt")
    normdata, ranges, minvals = auto_norm(datingdatamat)
    m, n = normdata.shape
    numtest = int(m * horatio)
    errorcount = 0.0
    for i in range(numtest):
        classifierresult = knn_classify(normdata[i, :], normdata[numtest:m, :], datinglabels[numtest:m], 3)
        print("Classifier: ", classifierresult, " Real answer:", datinglabels[i])
        if classifierresult != datinglabels[i]:
            errorcount += 1.0
    print("the error rate is: %f" % (errorcount / numtest))


if __name__ == "__main__":
    # group, labels = create_dataset()
    # testlabel = knn_classify([0, 0], group, labels, 3)
    # print(testlabel)
    Xdata, Ylabel = load_dataset("datingTestSet2.txt")
    print(Xdata)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Xdata[:, 0], Xdata[:, 1], 15.0 * np.array(Ylabel), 15.0 * np.array(Ylabel))
    plt.show()
    # normdata, ranges, minvals = auto_norm(Xdata)
    # print("N", normdata, "R", ranges, "M", minvals)

    datingclass_test()
