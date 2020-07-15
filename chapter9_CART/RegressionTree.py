# @Time : 2019/8/1 17:21 
# @Author : HXT
# @File : RegressionTree.py 
# @Software: PyCharm
# ======================================================================================================================
# Classification And Regression Tree
import numpy as np


# 树结点定义
class TreeNode():
    def __init__(self, curfeature, val, right, left):
        featuretosplit = curfeature
        valueofsf = val
        righttreee = right
        lefttree = left


#
def load_dataset(filename):
    datamat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split("\t")
        floatline = list(map(float, curline))  # map返回迭代器
        datamat.append(floatline)
    datamat = np.array(datamat)
    return datamat


# 叶结点模型
def regleaf(dataset):
    return np.mean(dataset[:, -1])


# 误差估计(均方总误差)
def regerr(dataset):
    return np.var(dataset[:, -1]) * dataset.shape[0]


# 二元分类法分割数据
def binsplit_dataset(dataset, feature, featurevalue):
    mat0 = dataset[dataset[:, feature] > featurevalue, :]
    mat1 = dataset[dataset[:, feature] <= featurevalue, :]

    return mat0, mat1


# 选择最优划分特征(特征==特征取值 划分)
def choose_bestfeature(dataset, leaftype, errtype, ops):
    tolE, toln = ops[0], ops[1]  # tols 允许的误差下降值,toln 切分的最少样本数
    m, n = dataset.shape

    if len(set(dataset[:, -1].reshape(1, m).tolist()[0])) == 1:  # 样本取值相同
        return None, leaftype(dataset)

    # 选取最优属性
    baseE = errtype(dataset)  # 评分,基于方差
    bestE = np.inf
    bestfeature, bestfvalue = 0, 0
    for feature in range(n - 1):  # 遍历所有属性
        for curfvalue in set(dataset[:, feature]):  # 遍历当前属性的所有取值
            mat0, mat1 = binsplit_dataset(dataset, feature, curfvalue)
            if (mat0.shape[0] < toln) or (mat1.shape[0] < toln):
                continue
            curE = errtype(mat0) + errtype(mat1)
            if curE < bestE:
                bestfeature = feature
                bestfvalue = curfvalue
                bestE = curE

    print(bestfeature, bestfvalue)
    # 预剪枝
    if baseE - bestE < tolE:  # 避免过拟合
        return None, leaftype(dataset)

    # mat0, mat1 = binsplit_dataset(dataset, bestfeature, bestfvalue)
    # if (mat0.shape[0] < toln) or (mat1.shape[0] < toln):
    #     return None, leaftype(dataset)

    return bestfeature, bestfvalue


def create_tree(dataset, leaftype=regleaf, errtype=regerr, ops=(1, 4)):
    feature, featurevalue = choose_bestfeature(dataset, leaftype, errtype, ops)
    if feature == None:
        return featurevalue  # 目标取值

    regressiontree = {}

    regressiontree["NodeFeature"] = feature
    regressiontree["NodeFeatureValue"] = featurevalue
    leftdataset, rightdataset = binsplit_dataset(dataset, feature, featurevalue)
    regressiontree["Ltree"] = create_tree(leftdataset, leaftype, errtype, ops)
    regressiontree["Rtree"] = create_tree(rightdataset, leaftype, errtype, ops)

    return regressiontree


# 判断对象是否是树形结构
def is_tree(obj):
    return (type(obj).__name__ == "dict")


#
def get_mean(tree):
    if is_tree(tree["Ltree"]):
        tree["Ltree"] = get_mean(tree["Ltree"])
    if is_tree(tree["Rtree"]):
        tree["Rtree"] = get_mean(tree["Rtree"])
    # 分支上都是叶结点
    return (tree["Ltree"] + tree["Rtree"]) / 2.0


# 剪枝
def after_pruning(tree, testdata):
    if testdata.shape[0] == 0:
        return get_mean(tree)
    # 用于向下递归的数据
    leftdataset, rightdataset = None, None
    if is_tree(tree["Ltree"]) or is_tree(tree["Rtree"]):
        leftdataset, rightdataset = binsplit_dataset(testdata, tree["NodeFeature"], tree["NodeFeatureValue"])
    # 向下递归
    if is_tree(tree["Ltree"]):
        tree["Ltree"] = after_pruning(tree["Ltree"], leftdataset)
    if is_tree(tree["Rtree"]):
        tree["Rtree"] = after_pruning(tree["Rtree"], rightdataset)

    # 回溯剪枝(分支上都是叶结点)
    if not is_tree(tree["Ltree"]) and not is_tree(tree["Rtree"]):
        leftdataset, rightdataset = binsplit_dataset(testdata, tree["NodeFeature"], tree["NodeFeatureValue"])
        # 未合并误差
        NoMergeE = np.sum(np.power(leftdataset[:, -1] - tree["Ltree"], 2)) + \
                   np.sum(np.power(rightdataset[:, -1] - tree["Rtree"], 2))
        # 合并误差
        treemean = (tree["Ltree"] + tree["Rtree"]) / 2.0
        MergeE = np.sum(np.power(testdata[:, -1] - treemean, 2))
        # 判断是否剪枝
        if MergeE < NoMergeE:
            print("merge")
            return treemean  # 设为叶结点
        else:
            return tree
    else:
        return tree


# 数据测试
# testmat = np.eye(4)
# mat0, mat1 = binsplit_dataset(testmat, 1, 0.5)
# print(mat0,"\n",mat1)


# 回归树测试
# mymat = load_dataset("ex00.txt")
# mytree = create_tree(mymat, ops=(0, 4))
# print(mytree)
#
# mymat = load_dataset("ex0.txt")
# mytree = create_tree(mymat)
# print(mytree)


mymat2 = load_dataset("ex2.txt")
mytree = create_tree(mymat2, ops=(0, 1))

mymat2test = load_dataset("ex2test.txt")
after_pruning(mytree, mymat2test)
print(mytree)
