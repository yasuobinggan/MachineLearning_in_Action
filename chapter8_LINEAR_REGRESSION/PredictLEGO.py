# @Time : 2019/7/23 16:05 
# @Author : HXT
# @File : PredictLEGO.py 
# @Software: PyCharm
# ======================================================================================================================
import numpy as np
from chapter8_LINEAR_REGRESSION.RIDGERegression import ridgeregression_test


# 交叉验证
# Xdata,Ydata列表格式 numcrossval 交叉验证的次数
def crossvaildation(Xdata, Ydata, numcrossval=10):
    m = len(Ydata)
    indexlist = range(m)  # 用于给数据创建下标
    errormat = np.zeros((numcrossval, 30))  # 保存误差结果

    for i in range(numcrossval):
        Xtrain, Ytrain = [], []
        Xtest, Ytest = [], []
        np.random.shuffle(numcrossval)

        for j in range(m):
            if j < 0.9 * m:
                Xtrain.append(Xdata[indexlist[j]])
                Ytrain.append(Ydata[indexlist[j]])
            else:
                Xtest.append(Xdata[indexlist[j]])
                Ytest.append(Ydata[indexlist[j]])

        theta = ridgeregression_test(np.array(Xtrain, Ytrain))

        for k in range(30):
            Xtestmat = np.array(Xtest)
            Ytestmat = np.array(Ytest)
            #.......
