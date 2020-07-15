# @Time : 2019/8/4 10:43 
# @Author : HXT
# @File : NaiveBayesForWatermelon.py 
# @Software: PyCharm
# ======================================================================================================================
# 简单的朴素贝叶斯分类器
# 围绕 NP难题
# 条件概率表的条件应该做表头(以概率为基础的模型)
# 基于《机器学习》周志华
import numpy as np


def load_dataset(filename):
    dataset = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split("\t")
        curlist = list(map(float, curline))
        dataset.append(curlist)

    return np.array(dataset)


# 获得先验概率
def get_priorpro(dataset):
    ProC = {}
    print("Prior Probability:")
    for curlabel in dataset[:, -1]:
        if curlabel not in ProC.keys():
            ProC[curlabel] = [0, 0]  # 前概率,后计数
        ProC[curlabel][1] += 1
    m = dataset.shape[0]
    N = len(ProC.keys())
    for curlabel in ProC.keys():
        ProC[curlabel][0] = (ProC[curlabel][1] + 1) / (m + N)
        print(curlabel, ProC[curlabel][0])

    # labels = set(dataset[:, -1])
    return ProC


# 属性取值对应的条件概率
# 基于多层字典的数据结构
# 适用多分类问题
def get_feature_conpro(dataset, feature, ProC):
    ProFea = {}
    for curlabel in ProC.keys():  # 分类标签为表头
        ProFea[curlabel] = {}
        curdataset = dataset[dataset[:, -1] == curlabel, :]  # 括号内是bool矩阵

        for curfea in feature.keys():  # 处理一个属性
            ProFea[curlabel][curfea] = {}
            if type(feature[curfea][1]).__name__ == "list":  # 离散属性
                Ni = len(feature[curfea][1])  # 属性取值数量
                for curfeavalue in feature[curfea][1]:  # 遍历所有属性取值
                    ele = []
                    cnt = curdataset[curdataset[:, curfea] == curfeavalue, :].shape[0]
                    pro = (cnt + 1) / (ProC[curlabel][1] + Ni)
                    ele.extend([pro, cnt])  # 前概率,后计数
                    ProFea[curlabel][curfea][curfeavalue] = ele

            else:  # 连续属性
                feamean = np.mean(curdataset[:, curfea])  # 取出一整列
                feavar = np.var(curdataset[:, curfea])  # 取出一整列
                for curfeavalue in set(list(dataset[:, curfea])):  # 遍历总数据集所有属性取值
                    ele = []
                    pro = 1 / (np.sqrt(2 * np.pi * feavar)) * \
                          np.exp(-((curfeavalue - feamean) ** 2) / (2 * feavar))
                    ele.extend([pro, 0])  # 前概率,后计数
                    ProFea[curlabel][curfea][curfeavalue] = ele

    return ProFea


# 对样本进行预测
def predict(sample, ProC, ProFea):
    tarpro = {}
    chooselabel = -1
    maxpro = -np.inf

    for curlabel in ProFea.keys():
        curtarpro = np.log(ProC[curlabel][0])
        for curfea in ProFea[curlabel].keys():
            # print(sample[curfea])
            curtarpro += np.log(ProFea[curlabel][curfea][sample[curfea]][0])
        tarpro[curlabel] = curtarpro
        if curtarpro > maxpro:
            maxpro = curtarpro
            chooselabel = curlabel
    print("Cursample Classify ans", tarpro)
    return chooselabel


# 模型对训练集的预测精度
def get_accuracy(dataset, ProC, ProFea):
    accuracy = 0
    for i in range(dataset.shape[0]):
        yhyp = predict(dataset[i, :], ProC, ProFea)
        if yhyp == dataset[i][-1]:
            accuracy += 1
    print("Accuracy:", accuracy / dataset.shape[0])


if __name__ == "__main__":
    dataset = load_dataset("Watermelon.txt")
    feature = {0: ("色泽", [0, 1, 2]), 1: ("根蒂", [0, 1, 2]), 2: ("敲声", [0, 1, 2]), 3: ("纹理", [0, 1, 2]),
               4: ("脐部", [0, 1, 2]), 5: ("触感", [0, 1]), 6: ("密度", -1), 7: ("含糖率", -1)}  # -1 标记连续属性
    ProC = get_priorpro(dataset)
    ProFea = get_feature_conpro(dataset, feature, ProC)
    print("NB Classifier", ProFea)
    get_accuracy(dataset, ProC, ProFea)
