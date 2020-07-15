# @Time : 2019/7/27 20:41 
# @Author : HXT
# @File : DecisionTree.py
# @Software: PyCharm
# ======================================================================================================================
# coding=utf-8
# ID3决策树
import numpy as np
import math
import operator
import sys
import pickle


# 创建数据集
def create_dataset():
    dataset = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    features = ["no surfacing", "flippers"]
    return dataset, features


# 预设两棵树
def retrieveTree(i):
    listoftrees = [{"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
                   {"no surfacing": {0: "no", 1: {"flippers": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}]
    return listoftrees[i]


# 计算香农熵(基于多分类问题)
def calc_shannonent(dataset):
    sumcount = len(dataset)
    labelcounts = {}
    # 处理数据(对字典的处理)
    for sample in dataset:
        curlabel = sample[-1]
        if curlabel not in labelcounts.keys():  # 初始化
            labelcounts[curlabel] = 0
        labelcounts[curlabel] += 1
    # 香农熵
    shannonent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key]) / sumcount
        shannonent -= prob * np.log2(prob)
    return shannonent


# 根据属性,属性取值划分数据集
def split_dataset(dataset, feature, curfvalue):
    retdataset = []  # 划分后的数据集
    for sample in dataset:
        if sample[feature] == curfvalue:  # 以删去当前属性对应的列构建子数据
            reducesample = sample[:feature]
            reducesample.extend(sample[feature + 1:])
            retdataset.append(reducesample)
    return retdataset


# 选择待划分属性
def choose_feature(dataset):
    numfeature = len(dataset[0]) - 1  # 特征数量
    baseentropy = calc_shannonent(dataset)

    bestgain, bestfeature = 0.0, -1
    for i in range(numfeature):
        # 数据处理
        fvaluelist = [example[i] for example in dataset]  # 当前属性的取值列表
        fvalueset = set(fvaluelist)  # 当前属性的取值集合
        # 划分后的信息熵
        newentropy = 0.0
        for curfvalue in fvalueset:
            subdataset = split_dataset(dataset, i, curfvalue)  # i为当前属性的标号
            prob = len(subdataset) / float(len(dataset))
            newentropy += prob * calc_shannonent(subdataset)
        # 信息增益
        curgain = baseentropy - newentropy
        # 选择最优
        if curgain >= bestgain:
            bestfeature = i
            bestgain = curgain

    return bestfeature


# 选择当前最多的标签(基于多数表决)
def majority_cnt(classlist):
    classcount = {}
    for curlabel in classlist:
        if curlabel not in classcount:
            classcount[curlabel] = 0
        classcount[curlabel] += 1
    maxlabel = -sys.maxsize
    maxvote = -sys.masize
    for curlabel in classcount:
        if classcount[curlabel] > maxvote:
            maxlabel = curlabel
            maxvote = classcount[curlabel]
    return maxlabel


# 递归构建决策树
def create_tree(dataset, features):
    classlist = [example[-1] for example in dataset]  # 数据集中的类别标签
    # 类别完全相同
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    # 遍历完所有特征,返回出现次数最多的类别(标签)
    if len(dataset[0]) == 1:
        return majority_cnt(classlist)
    # 选择最优特征
    bestfeature = choose_feature(dataset)
    bestfname = features[bestfeature]
    # 构建当前结点
    mytree = {bestfname: {}}  # 字典类型

    del (features[bestfeature])  # 划分完数据集后删去这个属性

    # 构建孩子结点
    bestfvaluelist = [sample[bestfeature] for sample in dataset]
    bestfvalueset = set(bestfvaluelist)
    for curvalue in bestfvalueset:
        # subfeatures = features[:]
        subfeatures = features.copy()  # 利用深复制完成回溯
        mytree[bestfname][curvalue] = create_tree(split_dataset(dataset, bestfeature, curvalue), subfeatures)  #
    return mytree


# 预测
def classify(tree, feature, sample):
    firstkey = next(iter(tree))
    child = tree[firstkey]
    featureid = feature.index(firstkey)
    for key in child:
        if sample[featureid] == key:
            if type(child[key]).__name__ == "dict":
                classlabel = classify(child[key], feature, sample)
            else:
                classlabel = child[key]
    return classlabel


# 存储数据
def store_tree(inputtree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputtree, fw)
    fw.close()


# 提取数据
def grab_tree(filename):
    fr = open(filename)
    return pickle.load(fr)


if __name__ == "__main__":
    dataset, features = create_dataset()
    print(features)

    # print(calc_shannonent(dataset))
    # dataset[0][-1] = "maybe"
    # print(calc_shannonent(dataset))
    # print(choose_feature(dataset))

    MYtree = create_tree(dataset, features)
    # print(MYtree)
    print(features)

    # mytree = retrieveTree(0)
    # print(classify(mytree, features, [1, 0]))
    # print(classify(mytree, features, [1, 1]))

    store_tree(MYtree, "ClassifierStorage.txt")
    grab_tree("ClassifierStorage.txt")
