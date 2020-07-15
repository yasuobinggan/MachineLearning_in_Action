# @Time : 2019/8/3 9:45 
# @Author : HXT
# @File : ModelTree.py 
# @Software: PyCharm
# ======================================================================================================================
# 叶结点是线性回归的决策树
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


# 线性回归
def linear_solve(dataset):
    m, n = dataset.shape
    X = np.ones((m, n))
    Y = np.ones((m, 1))
    # 读入数据
    X[:, 1:n] = dataset[:, 0:n - 1]
    Y[:] = dataset[:, -1].reshape(m, 1)
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
    return theta, X, Y


# 模型树叶结点(返回线性模型的参数)
def modelleaf(dataset):
    theta, X, Y = linear_solve(dataset)  # 模型树特征
    return theta


# 模型树误差评分
def modelerr(dataset):  # 基于当前数据集
    theta, X, Y = linear_solve(dataset)  # 模型树特征
    Yhyp = np.dot(X, theta)
    return np.sum((Y - Yhyp) ** 2)


# 二元分类法分割数据
# 左大右小
def binsplit_dataset(dataset, feature, featurevalue):
    mat0 = dataset[dataset[:, feature] > featurevalue, :]
    mat1 = dataset[dataset[:, feature] <= featurevalue, :]

    return mat0, mat1


# 选择最优划分特征(特征==特征取值 划分)
def choose_bestfeature(dataset, leaftype, errtype, ops):
    tolE, toln = ops[0], ops[1]  # tols 允许的误差下降值,toln 切分的最少样本数
    m, n = dataset.shape

    # 当前样本分类相同,停止划分
    if len(set(dataset[:, -1].reshape(1, m).tolist()[0])) == 1:
        return None, leaftype(dataset)

    # 选取最优属性
    baseE = errtype(dataset)  # 评分,基于方差
    bestE = np.inf
    bestfeature, bestfvalue = 0, 0
    for feature in range(n - 1):  # 遍历所有属性

        for curfvalue in set(list(dataset[:, feature])):  # 遍历当前属性的所有取值
            mat0, mat1 = binsplit_dataset(dataset, feature, curfvalue)
            if (mat0.shape[0] < toln) or (mat1.shape[0] < toln):
                continue
            curE = errtype(mat0) + errtype(mat1)
            if curE < bestE:
                bestfeature = feature
                bestfvalue = curfvalue
                bestE = curE

    # print(bestfeature, bestfvalue)
    # 预剪枝
    if baseE - bestE < tolE:  # 避免过拟合
        return None, leaftype(dataset)

    # mat0, mat1 = binsplit_dataset(dataset, bestfeature, bestfvalue)
    # if (mat0.shape[0] < toln) or (mat1.shape[0] < toln):
    #     return None, leaftype(dataset)

    return bestfeature, bestfvalue


#
def create_tree(dataset, leaftype=modelleaf, errtype=modelerr, ops=(1, 4)):
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


# 后剪枝
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


# 查看模型树
# mymat = load_dataset("exp2.txt")
# mytree = create_tree(mymat, modelleaf, modelerr, (1, 10))
# print(mytree)

'''预测部分'''


# 回归树预测值
def regtree_eval(model, indata):
    return np.float(model)


# 模型树预测值
def modeltree_eval(model, indata):
    n = indata.shape[1]
    X = np.ones((1, n + 1))
    X[:, 1:n + 1] = indata
    return np.float(np.dot(X, model))  # model是线性回归的参数


# 对树模型进行预测
def treeforecast(tree, indata, eval=modeltree_eval):
    if not is_tree(tree):
        return eval(tree, indata)
    # 判断分支
    if indata[tree["NodeFeature"]] > tree["NodeFeatureValue"]:
        return treeforecast(tree["Ltree"], indata, eval)
    else:
        return treeforecast(tree["Rtree"], indata, eval)


# 对测试集预测
def create_forecast(tree, testdata, eval=modeltree_eval):
    m = testdata.shape[0]
    n = testdata.shape[1]
    Yhyp = np.zeros((m, 1))
    for i in range(m):
        Yhyp[i, :] = treeforecast(tree, testdata[i, :].reshape(1, n), eval)
    return Yhyp


trainingset = load_dataset("bikeSpeedVsIq_train.txt")
testset = load_dataset("bikeSpeedVsIq_test.txt")
m, n = testset.shape
testset0 = testset[:, 0].reshape(m, 1)

# 回归树
retree = create_tree(trainingset, leaftype=regleaf, errtype=regerr, ops=(1, 20))
print("Regression Tree", retree)
Yhyp = create_forecast(retree, testset0, eval=regtree_eval)
print("regression tree accuracy", np.corrcoef(Yhyp, testset[:, 1], rowvar=0)[0, 1])

# 模型树
motree = create_tree(trainingset, leaftype=modelleaf, errtype=modelerr, ops=(1, 20))
print("Model Tree", motree)
Yhyp = create_forecast(motree, testset0, eval=modeltree_eval)
print("model tree accuracy", np.corrcoef(Yhyp, testset[:, 1], rowvar=0)[0, 1])
