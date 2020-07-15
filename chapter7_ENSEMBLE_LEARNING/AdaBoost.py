# @Time : 2019/8/14 16:08 
# @Author : HXT
# @File : AdaBoost.py.py
# @Software: PyCharm
# ======================================================================================================================
# AdaBoost
# stump 树桩
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def load_simpledata():
    datamat = np.mat([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classlabels = [1.0, 1.0, -1.0, -1.0, 1.0]  # 标记成正负二分类
    return datamat, classlabels


def load_data(filename):
    fr = open(filename)
    datamat, classlabels = [], []
    for line in fr.readlines():
        linelist = line.strip().split()  # 默认字符串形式 split默认参数以空格分割字符串
        curdata = []
        for i in range(len(linelist) - 1):
            ele = float(linelist[i])
            curdata.append(ele)
        datamat.append(curdata)
        classlabels.append(float(linelist[-1]))

    datamat = np.mat(datamat)
    return datamat, classlabels


# 依据决策树桩(现成的决策树桩)对数据分类
# 返回预测值
# datamat: 数据, feature: 特征, threshineq: 判别类型(大于或小于), threshval: 阈值
def stumpclassify(datamat, feature, threshineq, threshval):
    m, n = datamat.shape
    retarray = np.ones((m, 1))  # 对属性这一列进行分类

    if threshineq == "lt":
        retarray[datamat[:, feature] <= threshval] = -1.0
    else:
        retarray[datamat[:, feature] > threshval] = -1.0
    return retarray


# 在加权数据集中循环,找最低错误率的决策树桩
# 建立最优决策树桩不同于建立决策树 采用暴力匹配(所有的属性,属性的相应取值)最优准确率的决策树桩
# 每个决策树桩即为弱学习器即可
# D: 分布(数据分布,样本分布)实现是使用权重矩阵
def buildstump(datamat, classlabels, D):
    lablemat = np.mat(classlabels).T
    m, n = datamat.shape
    numsteps = 10  # 遍历连续属性时的数量
    # 获取最优解的两个变量
    beststump = {}
    bestclass = np.mat(np.zeros((m, 1)))

    minerror = np.inf  # 获取最低误差权重
    for i in range(n):  # 遍历所有属性
        rangemin = np.min(datamat[:, i])
        rangemax = np.max(datamat[:, i])
        stepsize = (rangemax - rangemin) / numsteps  # 遍历步长
        for j in range(-1, numsteps + 1):  # 遍历所有属性取值
            # 二分类问题(属性取值)
            for inequal in ["lt", "gt"]:  # 大于和小于的情况均遍历，lt:Less than  gt:greater than
                threshval = (rangemin + j * stepsize)  # 阈值
                predictedval = stumpclassify(datamat, i, inequal, threshval)  # 预测值
                # 计算误差矩阵(错误权重: 1, 正确权重: 0)
                errarr = np.mat(np.ones((m, 1)))
                errarr[predictedval == lablemat] = 0  # TODO 代码
                # 误差权重
                weighterror = D.T * errarr  # (1,m * m,1) = 1,1

                # print("feature: %d, thresh inequal: %s, thresh: %.2f, weighterror: %.3f"\
                #       % (i, inequal, threshval, weighterror))

                # 是否是最优决策树
                if weighterror < minerror:
                    minerror = weighterror
                    bestclass = predictedval.copy()
                    beststump["feature"] = i
                    beststump["ineq"] = inequal
                    beststump["thresh"] = threshval

    return minerror, beststump, bestclass  # 误差权重,基分类器,基分类器的分类


# AdaBoost
# 返回分类器
# datamat: 数据, classlabels: 标签, T: 迭代上限
def adaboost(datamat, classlabels, T):
    m, n = datamat.shape

    baseclassifiers = []  # 基分类器的集合
    aggclassest = np.mat(np.zeros((m, 1)))  #
    D = np.mat(np.ones((m, 1)) / m)

    for i in range(T):
        error, beststump, curclass = buildstump(datamat, classlabels, D)
        print("D:", D.T)
        alpha = np.float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 学习器权重
        beststump["alpha"] = alpha
        baseclassifiers.append(beststump)  # 加入基分类器集合
        print("Classest:", curclass.T)
        # 计算分布
        expon = np.multiply(-1 * alpha * np.mat(classlabels).T, curclass)  # 分布中的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 计算误差
        aggclassest += alpha * curclass  # 累加分类器的分类结果
        print("aggclassset:", aggclassest.T)
        aggerrors = np.multiply(np.sign(aggclassest) != np.mat(classlabels).T, np.ones((m, 1)))  # 记录误差
        errorrate = aggerrors.sum() / m
        print("total error_rate:", errorrate)
        print("***************************************************")

        if errorrate == 0.0:
            break

    return baseclassifiers, aggclassest


# ada分类
def ada_classify(datatoclass, classifiers):
    datamat = np.mat(datatoclass)
    m, n = datamat.shape
    aggclassest = np.mat(np.zeros((m, 1)))
    # 线性组合求和
    for i in range(len(classifiers)):
        classest = stumpclassify(datamat, classifiers[i]["feature"], classifiers[i]["ineq"], classifiers[i]["thresh"])
        aggclassest += classifiers[i]["alpha"] * classest

    return np.sign(aggclassest)


# 绘制roc曲线
# predstrengths: 分类器的预测强度
def plot_roc(predstrengths, classlabels):
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    cur = (1.0, 1.0)  # 绘制光标的位置
    ysum = 0.0  # 用于计算AUC
    numposclas = np.sum(np.array(classlabels) == 1.0)  # 正例数目
    ystep = 1 / float(numposclas)  # y轴步长
    xstep = 1 / float(len(classlabels) - numposclas)  # x轴步长
    sortedindicies = np.argsort(predstrengths)

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for index in sortedindicies.tolist()[0]:
        # 多一个TP向y轴移动一步
        if classlabels[index] == 1.0:
            delx = 0
            dely = ystep
        # 多一个FP向x轴移动一步
        else:
            delx = xstep
            dely = 0
            ysum += cur[1]
        # 绘制ROC
        ax.plot([cur[0], cur[0] - delx], [cur[1], cur[1] - dely], c="b")
        # 更新绘制光标位置
        cur = (cur[0] - delx, cur[1] - dely)

    ax.plot([0, 1], [0, 1], "b--")
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties=font)
    plt.xlabel('假阳率', FontProperties=font)
    plt.ylabel('真阳率', FontProperties=font)
    ax.axis([0, 1, 0, 1])
    # 计算AUC
    print('AUC面积为：', ysum * xstep)
    plt.show()


if __name__ == "__main__":
    # datamat, classlabels = load_simpledata()
    # D = np.mat(np.ones((5, 1)) / 5)
    # minerror, beststump, bestclass = buildstump(datamat, classlabels, D)
    # print(minerror, beststump, bestclass)
    # classifiers = adaboost(datamat, classlabels, 30)
    # print(classifiers)

    # print(ada_classify([0, 0], classifiers))
    # print(ada_classify([[5, 5], [0, 0]], classifiers))

    datamat, classlabels = load_data("horseColicTraining2.txt")
    classifiers, aggclassest = adaboost(datamat, classlabels, 10)
    plot_roc(aggclassest.T, classlabels)

    testdatamat, testclasslabels = load_data("horseColicTest2.txt")
    predict10 = ada_classify(testdatamat, classifiers)

    errarr = np.mat(np.ones((67, 1)))
    print(np.sum(errarr[predict10 != np.mat(testclasslabels).T]))
