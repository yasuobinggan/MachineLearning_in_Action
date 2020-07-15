# @Time : 2019/7/14 19:40 
# @Author : HXT
# @File : PredictAbalone.py 
# @Software: PyCharm
# ======================================================================================================================
# 预测鲍鱼年龄 基于局部加权线性回归
import numpy as np
import matplotlib.pyplot as plt
import chapter8_LINEAR_REGRESSION.Util as util
import chapter8_LINEAR_REGRESSION.LWLinearRegression as LWLR

# 加载数据进行预测
abX, abY = util.load_dataset("abalone.txt")
# print(np.array(abY))
# print(np.array(abY).shape)
Yhyp1 = LWLR.lwlrtest(abX[0:99], abX[0:99], abY[0:99], 0.1)
Yhyp2 = LWLR.lwlrtest(abX[0:99], abX[0:99], abY[0:99], 1)
Yhyp3 = LWLR.lwlrtest(abX[0:99], abX[0:99], abY[0:99], 10)
# print(Yhyp1.shape)

# 输出误差
abY = np.array(abY)
print("k==0.1 error", util.rss_error(abY[0:99], Yhyp1))
print("k==1 error", util.rss_error(abY[0:99], Yhyp2))
print("k==10 error", util.rss_error(abY[0:99], Yhyp3))
