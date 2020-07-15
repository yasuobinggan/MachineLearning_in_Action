# @Time : 2019/8/13 20:51 
# @Author : HXT
# @File : KNNForDigit.py 
# @Software: PyCharm
# ======================================================================================================================
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import operator


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


# 转换成数字向量
def img2vector(filename):
    returnvec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvec[0, 32 * i + j] = int(linestr[j])
    return returnvec


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
    # 计算距离
    diffmat = (np.tile(inX, (m, 1)) - Xdata) ** 2
    distances = np.sum(diffmat, axis=1) ** 0.5  # 返回一个无维数组
    # 排序
    sorteddistanceindicies = np.argsort(distances, axis=0)
    # 投票
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
def handwritingclass_test():
    # 加载训练集
    hwlabels = []
    trainingfilelist = listdir("trainingDigits")
    m = len(trainingfilelist)
    trainingmat = np.zeros((m, 1024))
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split(".")[0]
        classnumstr = int(filestr.split("_")[0])  # 取出真实标记
        hwlabels.append(classnumstr)  # 存入真实标记
        trainingmat[i, :] = img2vector("trainingDigits/%s" % filenamestr)  # 存入训练向量
    # 处理测试集
    testfilelist = listdir("testDigits")
    errorcount = 0.0
    numtest = len(testfilelist)
    for i in range(numtest):
        filenamestr = testfilelist[i]
        filestr = filenamestr.split(".")[0]  #
        classnum = int(filestr.split("_")[0])  # 当前测试集的分类
        vectortest = img2vector("testDigits/%s" % filenamestr)  # img->vector
        classifierresult = knn_classify(vectortest, trainingmat, hwlabels, 3)  # knn分类
        print("Classifier: %d, Real answer: %d" % (classifierresult, classnum))
        if classifierresult != classnum:
            errorcount += 1.0
    print("number of errors is %d" % (errorcount))
    print("error rate %f" % (errorcount / numtest))


if __name__ == "__main__":
    # testvec = img2vector("testDigits/0_13.txt")
    # print(testvec[0, 0:31], testvec[0, 32:63])
    handwritingclass_test()
