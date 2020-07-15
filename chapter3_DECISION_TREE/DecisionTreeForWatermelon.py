# @Time : 2019/7/25 20:48 
# @Author : HXT
# @File : DecisionTreeForWatermelon.py
# @Software: PyCharm
# ======================================================================================================================
# 创建一个(动态)决策树 以属性作为结点构建一颗树
# 数据集完整
# 基于《机器学习》周志华

import numpy as np
from collections import *
import sys, os
import math
import time


# 树结点
class Node():
    def __init__(self):
        self.flag = -1  # 标记 # 非叶:-1 叶:0 1
        self.fid = -1  # 当前结点的属性标号
        self.child = {}  # 属性取值:Node实例


def load_data(filename):
    fr = open(filename, "r")
    trainingset = []
    for line in fr.readlines():
        curline = line.strip().split("\t")

        sample = []
        for ele in curline:
            sample.append(int(ele))
        trainingset.append(sample)
    # print(trainingset)
    return np.array(trainingset)


# 对矩阵中的标签进行记录
def count_label(trainingset):
    cnt1, cnt0 = 0, 0
    for i in range(trainingset.shape[0]):
        if trainingset[i, -1] == 1:
            cnt1 += 1
        else:
            cnt0 += 1
    return cnt1, cnt0


# 信息熵(基于二分类问题)
def get_ent(trainingset):
    cnt1, cnt0 = count_label(trainingset)
    ele1 = cnt1 / trainingset.shape[0]
    ele0 = cnt0 / trainingset.shape[0]
    if ele1 != 0:
        ele1 = ele1 * np.log2(ele1)
    if ele0 != 0:
        ele0 = ele0 * np.log2(ele0)
    return -(ele1 + ele0)


# 信息增益 => 增益率(IV)
def get_gain(key, value, trainingset):
    sum = 0  # 分支的信息熵和
    # IV = 0  #
    for feavalue in value[1]:
        curtrainingset = trainingset[trainingset[:, key] == feavalue, :]  # 根据对应的取值划分数据集
        if len(curtrainingset) == 0:  # 防止0除错误
            sum += 0
            # IV += 0
        else:
            sum += curtrainingset.shape[0] / trainingset.shape[0] * get_ent(curtrainingset)
            # IV += (curtrainingset.shape[0] / trainingset.shape[0]) * \
            #       (np.log2(curtrainingset.shape[0] / trainingset.shape[0]))
    return sum  # , -IV


# 获取最优结点
def get_feature(trainingset, feature, featureset):
    maxkey = -1
    maxgain = -sys.maxsize
    gain = get_ent(trainingset)
    for key, value in feature.items():
        if key not in featureset:  # 关键标记
            curgain = get_gain(key, value, trainingset)
            # curgain, IV = get_gain(key, value, trainingset)
            if (gain - curgain) >= maxgain:  # if (gain - curgain) / IV > maxgain:
                maxkey = key
                maxgain = (gain - curgain)  # maxgain = (gain - curgain) / IV
    return maxkey


# ID3 => C4.5
def construct_tree(trainingset, feature, featureset):
    curfeature = []
    for curkey in feature.keys():
        if curkey not in featureset:
            curfeature.append(curkey)
    curfeature = np.array(curfeature)

    node = Node()

    # 在对数据进行判断中使用0,1标签的特性
    # 当前数据集的标签倾向
    labelflag = np.sum(trainingset[:, -1]) / trainingset.shape[0]

    # 再划分数据集无意义(trainingset中样本属于同一个类别)
    if labelflag == float(trainingset[0, -1]):
        node.flag = trainingset[0, -1]
        return node

    trainingsetBfeature = trainingset[:, curfeature]
    # 再划分属性无意义(feature = NULL of trainingset在feature上取值相同)
    if len(curfeature) == 0 or np.linalg.matrix_rank(trainingsetBfeature) == 1:
        if labelflag >= 0.5:
            node.flag = 1
        else:
            node.flag = 0
        return node

    # 获取最优属性
    fid = get_feature(trainingset, feature, featureset)
    node.fid = fid  # 补充当前结点信息

    featureset.add(fid)  # 当前路径不能再选择
    # 构建孩子结点
    for feavalue in feature[fid][1]:
        curtrainingset = trainingset[trainingset[:, fid] == feavalue, :]
        # curtrainingset为空 为孩子结点赋上trainingset的取值倾向
        if curtrainingset.shape[0] == 0:
            node.child[feavalue] = Node()
            if labelflag >= 0.5:
                node.child[feavalue].flag = 1
            else:
                node.child[feavalue].flag = 0
        # curtrainingset不为空
        else:
            node.child[feavalue] = construct_tree(curtrainingset, feature, featureset)
    featureset.remove(fid)  # 回溯

    return node


# 后剪枝
def after_pruning(root):
    pass


fcount = 0  # 记录树结点数量


# 对决策树遍历查看结果
def test_travel(node, feature):
    global fcount
    if node.flag == 0:
        # print(False)
        return
    if node.flag == 1:
        # print(True)
        return
    else:
        if type(node).__name__ == "Node":
            fcount += 1
        print(feature[node.fid][0], end="\t")
        for curchild in node.child.values():  # 输出孩子结点
            if curchild.flag == 0 or curchild.flag == 1:
                print(curchild.flag, end="\t")
            elif curchild.fid != -1:
                print(feature[curchild.fid][0], end="\t")
        print()
        for curchild in node.child.values():  # 向下递归
            test_travel(curchild, feature)


# 对单个样本预测
def predict(sample, node):
    while (node.flag != 1 and node.flag != 0):
        node = node.child[sample[node.fid]]
    return node.flag


if __name__ == "__main__":
    # 初始化问题数据
    trainingset = load_data("WatermelonTrain.txt")
    testset = load_data("WatermelonTest.txt")
    feature = {0: ("色泽", [0, 1, 2]), 1: ("根蒂", [0, 1, 2]), 2: ("敲声", [0, 1, 2]), 3: ("纹理", [0, 1, 2]),
               4: ("脐部", [0, 1, 2]), 5: ("触感", [0, 1])}
    featureset = set()

    # 构建决策树
    st = time.clock()
    root = construct_tree(trainingset, feature, featureset)
    ed = time.clock()

    # 遍历测试
    test_travel(root, feature)
    print("构建决策树时间", ed - st)
    print("FCOUNT", fcount)

    # 后剪枝
    # root = after_pruning(root)
    # test_travel(root,feature)

    # 当前模型对训练集的预测准确率
    print("******************验证集预测准确率******************")
    accuracy = 0
    for i in range(testset.shape[0]):
        curlabel = predict(testset[i, :], root)
        print("curlabel:", curlabel, "targetlabel:", testset[i, -1])
        if curlabel == testset[i, -1]:
            accuracy += 1
    accuracy /= testset.shape[0]
    print(accuracy)
