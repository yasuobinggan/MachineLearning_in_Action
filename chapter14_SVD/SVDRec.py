import numpy as np
import matplotlib.pyplot as plt


def load_data():
    return np.mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
                   [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
                   [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
                   [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
                   [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
                   [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
                   [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
                   [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
                   [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
                   [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
                   [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])
    # return np.mat([[2, 0, 0, 4, 4],
    #                [5, 5, 5, 3, 3],
    #                [2, 4, 2, 1, 2]])


'''
    相似度计算函数
    euclid_sim(inA, inB)
    pears_sim(inA, inB)
    cos_sim(inA, inB)
'''


# 欧式距离
def euclid_sim(inA, inB):
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


# 皮尔逊相关系数
def pears_sim(inA, inB):
    if len(inA) < 3:  # 硬编码基于这个当前问题
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=False)[0, 1]  # [0,1]的位置为inA,inB的相关系数


# 余弦相似度
def cos_sim(inA, inB):
    num = np.float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# 标准估计方法
# 利用已评分物品计算与未评分物品的相似度(距离)估计未评分物品的相似度
# 给定相似度计算方法的条件下,用户对物品的评估
# 参数: datamat 数据, user 当前用户, simmethod 计算相似度的方式, item
# 返回对当前物品的估计评分
def standest(datamat, user, simmethod, item):
    m, n = datamat.shape
    simtotal = 0.0
    ratsimtotal = 0.0
    for j in range(n):  # 遍历当前用户对应的所有物品
        userrating = datamat[user, j]  # 评分值
        if userrating == 0:  # 评分值为0,跳过当前物品
            continue
        # 当前用户(一个)未评分物品列 与 当前用户(一个)已评分物品列 中物品都被评分的用户列表
        overlap = np.nonzero(np.logical_and(datamat[:, item].A > 0, datamat[:, j].A > 0))[0]  # overlap是用户
        # 无用户同时对两个物品评分
        if len(overlap) == 0:
            similarity = 0
        # 有>=1个用户同时对两个物品评分
        else:
            similarity = simmethod(datamat[overlap, item], datamat[overlap, j])
        # print("the %d and %d similarity is: %f" % (item, j, similarity))
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        ratsimtotal += similarity * userrating  # 未评分物品的评分 += 当前(评分与未评分)相似度 * 评分物品的评分
        simtotal += similarity  # 总相似度 += 当前相似度
    if simtotal == 0:
        return 0
    else:
        # 通过除以所有的评分和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
        return ratsimtotal / simtotal


# 先用svd降维再用standest计算相似度
def svdest(datamat, user, simmethod, item):
    m, n = dataset.shape
    simtotal = 0.0
    ratsimtotal = 0.0
    U, Sigma, VT = np.linalg.svd(datamat)  # Sigma是一个无维数组
    Sig4 = np.mat(np.eye(4) * Sigma[:4])  # 利用Sigma前4个数据建立一个对角矩阵
    # 对于物品来说人可以当做特征
    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品（物品的4个主要特征）
    xformeditems = datamat.T * U[:, :4] * np.linalg.pinv(Sig4)

    for j in range(n):
        userrating = datamat[user, j]
        if userrating == 0:
            continue
        # 计算相似度
        similarity = simmethod(xformeditems[item, :].T, xformeditems[j, :].T)
        print("the %d and %d similarity is %f" % (item, j, similarity))
        ratsimtotal += similarity * userrating  # 未评分物品的评分 += 当前(评分与未评分)相似度 * 评分物品的评分
        simtotal += similarity  # 总相似度 += 当前相似度
    if simtotal == 0:
        return 0
    else:
        # 通过除以所有的评分和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
        return ratsimtotal / simtotal


# 推荐引擎
def recommend(datamat, user, N=3, simmethod=cos_sim, estmethod=svdest):
    unrateditems = np.nonzero(datamat[user, :].A == 0)[1]  # 寻找给定用户未评分的物品
    if len(unrateditems) == 0:  # 全部评分,退出函数
        return "you rated everything"
    itemscores = []  # 物品的编号和评分值(元素是元组)
    for item in unrateditems:
        estimatedscore = estmethod(datamat, user, simmethod, item)  # 返回评分
        itemscores.append((item, estimatedscore))
    # 逆序排序,根据评分从大到小排序取前N个进行推荐
    return sorted(itemscores, key=lambda jj: jj[1], reverse=True)[:N]


if __name__ == "__main__":
    dataset = np.mat(load_data())
    print("基于余弦相似度的评分", recommend(dataset, 2))
    print("基于欧式距离的评分", recommend(dataset, 2, simmethod=euclid_sim))
    print("基于相关系数的评分", recommend(dataset, 2, simmethod=pears_sim))

    # U, Sigma, VT = np.linalg.svd(load_data())
