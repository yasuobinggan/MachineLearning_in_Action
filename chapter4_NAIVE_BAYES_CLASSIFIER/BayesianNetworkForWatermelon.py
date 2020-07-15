# @Time : 2019/8/10 12:01 
# @Author : HXT
# @File : BayesianNetworkForWatermelon.py 
# @Software: PyCharm
# ======================================================================================================================
# K2自学习贝叶斯网络
# 贝叶斯网络推断依据概率论链式规则
# 多少个分类标签创建几个贝叶斯网络
# 基于《机器学习》周志华
import numpy as np


def load_dataset(filename):
    dataset, label = [], []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split("\t")
        curlist = list(map(float, curline))
        label.append(curlist[-1])
        del (curlist[-1])
        dataset.append(curlist)

    m = len(dataset)
    return np.array(dataset), np.array(label).reshape(m, 1)


# 数据预处理
dataset, label = load_dataset("WatermelonNOM.txt")  # 处理离散数据
features = {0: ("色泽", [0, 1, 2]), 1: ("根蒂", [0, 1, 2]), 2: ("敲声", [0, 1, 2]), 3: ("纹理", [0, 1, 2]),
            4: ("脐部", [0, 1, 2]), 5: ("触感", [0, 1])}
curfeatures = features


# 二级表项
class CPtableI():
    def __init__(self, cnt, pro):
        self.cnt = cnt
        self.pro = pro


# 一级表项
class CPtableH():
    def __init__(self, sumcnt):
        self.sumcnt = sumcnt
        self.curCPtableI = {}


# k2算法的预处理
def pre_process():
    featurefather = {}  # 记录属性的父结点
    for curfeature in curfeatures.keys():
        fathers = []
        for curfather in curfeatures.keys():
            if curfeature != curfather:
                fathers.append(curfather)
            else:
                break
        featurefather[curfeature] = fathers
    return featurefather


# 建表
def build_cptable(curfeature, comfeaturefather, index, curcomvalue, CPtable):
    if index == len(comfeaturefather):
        # 建立一级表项
        curcomvalue = tuple(curcomvalue)  # 列表不能做键名
        CPtable[curcomvalue] = CPtableH(0)
        # 建立二级表项
        for curfeaturevalue in curfeatures[curfeature][1]:
            CPtable[curcomvalue].curCPtableI[curfeaturevalue] = CPtableI(0, 0)
        return

    for curfeaturevalue in curfeatures[comfeaturefather[index]][1]:
        curcomvalue.append(curfeaturevalue)
        build_cptable(curfeature, comfeaturefather, index + 1, curcomvalue, CPtable)
        curcomvalue.pop()
        # curcomvalue.remove(curfeaturevalue)# 数值有重复部分不能使用remove


# 生成对应属性的条件概率表
def generate_nij_nijk(curfeature, comfeaturefather):
    CPtable = {}
    index = 0
    curcomvalue = []  # 当前路径上的条件取值
    curcomvalue.clear()
    # 建表
    build_cptable(curfeature, comfeaturefather, index, curcomvalue, CPtable)
    # 表中填充数据
    for sample in dataset:
        convalue = []
        for i in comfeaturefather:  # 取出条件概率取值
            convalue.append(sample[i])
        convalue = tuple(convalue)
        CPtable[convalue].sumcnt += 1
        CPtable[convalue].curCPtableI[sample[curfeature]].cnt += 1

    return CPtable


def cal_factor1(nij, ri):
    factor = 1
    st = nij + ri - 1
    while nij > 0:
        factor *= st
        st -= 1
        nij -= 1
    return factor


def cal_factor2(nijk):
    factor = 1
    while nijk > 0:
        factor *= nijk
        nijk -= 1
    return factor


def get_score(curfeature, comfeaturefather):
    CPtable = generate_nij_nijk(curfeature, comfeaturefather)
    score = 0.0
    # TODO
    for convalue in CPtable.keys():
        curscore = 1
        for value in CPtable[convalue].curCPtableI.keys():
            factor2 = cal_factor2(CPtable[convalue].curCPtableI[value].cnt)
            if factor2 != 0:
                curscore *= factor2
        curscore *= 1 / cal_factor1(CPtable[convalue].sumcnt, len(curfeatures[curfeature][1]))
        score += np.log(curscore)
    return score


# k2自生成贝叶斯网络
# 先验知识feature序列,limit父结点数量上限
def k2(featurefather, limit):
    BN = {}  # 字典类型
    sumscore = 0
    FF = featurefather.copy()

    for curfeature in curfeatures.keys():  # 遍历所有属性
        Pold = -np.inf  # 设置old评分
        OKToProcced = True
        curfeaturefather = FF[curfeature]  # 当前可能的父结点属性list
        comfeaturefather, bestfeaturefather = [], []  # 最佳父结点属性list

        while OKToProcced and len(bestfeaturefather) < limit and len(curfeaturefather) > 0:
            curfather = curfeaturefather.pop(0)  # 当前弹出父属性结点value
            comfeaturefather.append(curfather)  # 填充当前解
            Pnew = get_score(curfeature, comfeaturefather)  # Pnew
            # if-else 显示爬山结构
            if Pnew > Pold:
                Pold = Pnew
                bestfeaturefather = comfeaturefather
                sumscore += Pold
            else:
                OKToProcced = False

        BN[curfeature] = bestfeaturefather
        print(curfeatures[curfeature][0], bestfeaturefather)
    return BN, sumscore


# 扰动当前顺序
def swap_order():
    global curfeatures
    tempfeatures = {}
    tempfeatureslist = list(curfeatures.keys())

    a, b = 0, 0
    while True:
        a = np.random.randint(0,5)
        b = np.random.randint(0,5)
        if a != b:
            break
    temp = tempfeatureslist[a]
    tempfeatureslist[a] = tempfeatureslist[b]
    tempfeatureslist[b] = temp

    for curfeature in tempfeatureslist:
        tempfeatures[curfeature] = curfeatures[curfeature]

    curfeatures = tempfeatures
    # print(curfeatures)
    # return curfeatures


T, a = 1000, 0.01


# 模拟退火
def simulated_annealing(limit):
    global T, a
    featurefather = pre_process()
    BNsa, maxscore = k2(featurefather, limit)
    t = T
    curscore = 0
    deltaE = 0

    while t > 0:
        swap_order()  # 扰动当前解
        featurefather = pre_process()
        # print(featurefather)
        curBN, curscore = k2(featurefather, limit)
        deltaE = curscore - maxscore
        if deltaE > 0:
            maxscore = curscore
            BNsa = curBN
        else:
            p = np.exp(-deltaE / t)
            r = np.random.random()
            if p > r:
                maxscore = curscore
                BNsa = curBN
        t *= a

    print(BNsa)
    return BNsa


# 训练贝叶斯网络
def train_BN(dataset, BN):
    pass


if __name__ == "__main__":
    # 获得贝叶斯网络
    featurefather = pre_process()
    limit = 5
    print("**********k2**********")
    BN = k2(featurefather, limit)

    print("**********simulated annealing**********")
    BNsa = simulated_annealing(limit)
    # {2: [], 0: [2], 1: [2, 0], 4: [2, 0, 1], 3: [2, 0], 5: [2, 0]}
    # {1: [], 2: [1], 4: [1, 2], 3: [1, 2, 4], 0: [1, 2, 4, 3], 5: [1, 2]}
    # {4: [], 3: [4], 0: [4, 3], 1: [4, 3, 0], 2: [4, 3], 5: [4, 3]}
    # {1: [], 4: [1], 3: [1, 4], 0: [1, 4, 3], 2: [1, 4], 5: [1, 4, 3, 0]}
    # 训练贝叶斯网络
    train_BN(dataset, BN)
