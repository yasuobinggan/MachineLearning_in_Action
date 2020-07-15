# @Time : 2019/7/23 14:50 
# @Author : HXT
# @File : Util.py
# @Software: PyCharm
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt


# 读取数据(元素是列表)
def load_dataset(filename):
    fr = open(filename, "r")
    datamat = []
    labelmat = []

    # 按行读取(注意切割数据)
    for line in fr.readlines():  # 列表中每一个元素是字符串
        curline = line.strip().split("\t")
        # 数据类型转换成float
        for i in range(len(curline)):
            curline[i] = float(curline[i])
        datamat.append(curline[:len(curline) - 1])
        # labelmat.append(curline[-1])# 一维列表
        labelmat.append(curline[len(curline) - 1:])  # 元素是列表
    return datamat, labelmat  # 返回列表


# 返回均方误差
def rss_error(Ymat, Yhyp):
    # 注意矩阵相乘结果为一阶阵的维度，和原始一阶阵的维度
    return np.sum((Ymat.flatten() - Yhyp.flatten()) ** 2)  # Ymat.flatten()展成一维数组


# 数据标准化: MAT numpy数组类型
def regularize(MAT):
    # MAT = np.array(data)
    MATmean = np.mean(MAT, axis=0)
    MATvar = np.var(MAT, axis=0)
    return (MAT - MATmean) / MATvar
