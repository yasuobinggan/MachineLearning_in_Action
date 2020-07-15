#  @Time  : 2019/9/6 10:57
#  @Author : HXT
#  @Site  :
#  @File  : Apriori.py
#  @Software: PyCharm
# ======================================================================================================================

def load_data():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 构建大小为1的所有候选项集的集合
def create_init(dataset):
    c1 = []  # 元素为集合
    # 遍历数据中的所有单个物品
    for transaction in dataset:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])
    c1.sort()
    return list(map(frozenset, c1))


# 在单元素集合的集合中选取满足条件的集合
# 参数: dataset数据集,ck候选项集列表,minsupport最小支持度
def scan_d(dataset, ck, minsupport):
    sscnt = {}
    # 遍历全体样本中的每一个元素,记录当前项集中(集合)元素出现的次数
    for tid in dataset:  # 当前数据集中元素
        for can in ck:  # 当前项集的所有元素
            if can.issubset(tid):
                if can not in sscnt:
                    sscnt[can] = 1
                else:
                    sscnt[can] += 1

    numitems = float(len(dataset))  # 获取样本中的元素个数,用于计算支持度
    retlist = []  # 满足最小支持度要求的集合
    supportdata = {}  # 单元素频繁项集对应的支持度

    # 遍历每一个单元素
    for key in sscnt:
        support = sscnt[key] / numitems  # 计算支持率
        # 选择符合条件的集合
        if support >= minsupport:
            retlist.insert(0, key)
        supportdata[key] = support  # 集合对应的支持度
    return retlist, supportdata


# 创建ck
# 基于数学的合并方式
# 参数: lk频繁项集,k项集元素个数
def apriori_gen(lk, k):
    lenlk = len(lk)
    retlist = []
    # TODO
    for i in range(lenlk):
        for j in range(i + 1, lenlk):
            l1 = list(lk[i])[:k - 2]
            l2 = list(lk[j])[:k - 2]
            l1.sort()
            l2.sort()
            if l1 == l2:  # 两个集合的前k-2个元素
                retlist.append(lk[i] | lk[j])  # 并集
    return retlist


# 主函数
def apriori(data, minsupport=0.5):
    c1 = create_init(data)
    dataset = list(map(set, data))
    l1, supportdata = scan_d(data, c1, minsupport)  # l1单元素的频繁项集
    L = [l1]  # l 集合的集合的集合
    k = 2
    while (len(L[k - 2]) > 0):  # 直到下一个大项集为空
        ck = apriori_gen(L[k - 2], k)
        lk, supportk = scan_d(dataset, ck, minsupport)  # 根据长度产生新的频繁项集
        L.append(lk)  # 更新最大的集合
        supportdata.update(supportk)  # 更新集合对应的支持度
        k += 1
    return L, supportdata  # supportdata用于计算关联规则


# 计算可信度,寻找满足最小可信度要求的规则
def calc_conf(freqset, h, supportdata, bigrulelist, minconf=0.7):
    prunedh = []  # 满足最小可信度要求的规则列表
    for conseq in h:
        conf = supportdata[freqset] / supportdata[freqset - conseq]
        if conf >= minconf:
            print(freqset - conseq, "-->", conseq, "conf:", conf)
            bigrulelist.append((freqset - conseq, conseq, conf))  # 前件,后件,可信度
            prunedh.append(conseq)
    return prunedh


# 从当前项集生成更多的规则
def rules_from_conseq(freqset, h, supportdata, bigrulelist, minconf=0.7):
    m = len(h[0])
    if (len(freqset) > (m + 1)):  # 是否可以移除大小为m的子集
        # 下一次迭代使用的列表
        hmp1 = apriori_gen(h, m + 1)  # 生成无重复元素集合
        hmp1 = calc_conf(freqset, hmp1, supportdata, bigrulelist, minconf)  # 计算生成规则的可信度
        if (len(hmp1) > 1):
            rules_from_conseq(freqset, hmp1, supportdata, bigrulelist)  #


# L:频繁项集列表,supportdata:频繁项集的支持度,minconf:最小可信度
def generate_rules(L, supportdata, minconf=0.7):
    bigrulelist = []  # 存放满足可信度的列表 # 元素为元组
    for i in range(1, len(L)):  # 从长度为2开始,按长度遍历频繁项集集合
        for freqset in L[i]:  # 取当前长度频繁项集的集合中的元素(集合)
            h1 = [frozenset([item]) for item in freqset]  # 提取频繁项中单个元素
            # print("h1", h1)
            if i > 1:  # 频繁项集元素数目超过2,考虑进一步合并
                rules_from_conseq(freqset, h1, supportdata, bigrulelist, minconf)
            else:  # 初始化最开始的可信度(第一层)
                calc_conf(freqset, h1, supportdata, bigrulelist, minconf)
    return bigrulelist


if __name__ == "__main__":
    data = load_data()
    # L:所有长度的频繁项集合, L[i]:i长度频繁项集合, L[i][j]:一个频繁项集
    L, supportdata = apriori(data, minsupport=0.5)
    # print(L)print(L[0])print(L[1])print(L[2])print(L[3])
    # print("L1:", L[1])
    # print(apriori_gen(L[1], 3))
    rules = generate_rules(L, supportdata, minconf=0.7)
    print("***********************************")
    print(rules)
