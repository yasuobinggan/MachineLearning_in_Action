# @Time : 2019/7/25 16:36 
# @Author : HXT
# @File : PredictHorseColic.py 
# @Software: PyCharm
# ======================================================================================================================
import numpy as np
import chapter5_LOGISTIC_REGRESSION.LogisticRegression as LR


# 包名应该有规范

def classify_vector(x, theta):
    yhyp = LR.sigmoid(np.dot(x, theta))
    if yhyp > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    # 处理训练数据
    frtrain = open("horseColicTraining.txt")
    trainingset, traininglabels = [], []
    for line in frtrain.readlines():
        curline = line.strip().split("\t")
        linearr = []
        for i in range(21):
            linearr.append(float(curline[i]))
        trainingset.append(linearr)
        traininglabels.append(curline[21])

    # 训练
    Xmat = np.array(trainingset)
    Ymat = np.array(traininglabels)
    # theta = LR.grad_desecnt(Xmat,Ymat,0.01,0.01)
    # theta = LR.stocgrad_descent(Xmat,Ymat,0.01)
    theta = LR.stocgrad_descent_ev(Xmat, Ymat, numiters=500)


    # 对测试数据进行预测
    errorcount = 0  # 记录错误测试样本数量
    numtest = 0.0  # 记录测试样本数量
    frtest = open("horseColicTest.txt")
    for line in frtest.readlines():
        curline = line.strip().split("\t")
        linearr = []
        for i in range(21):
            linearr.append(float(curline[i]))
        curx = np.array(linearr)
        if int(classify_vector(curx.reshape(1,curx.size).astype(np.float), theta)) != int(curline[21]):
            errorcount += 1
        numtest += 1.0

    # 计算误差
    errorrate = (float(errorcount)/numtest)
    print("error rate of testset:",errorrate)
    return errorrate

# 进行多次测试(训练有随机成分在)
def multi_test():
    numiters = 10
    errorsum = 0.0

    for i in range(numiters):
        errorsum += colic_test()
    print("%d average error %f "%(numiters,float(errorsum/numiters)))

multi_test()