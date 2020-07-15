# @Time : 2019/8/10 11:32 
# @Author : HXT
# @File : BAYES.py 
# @Software: PyCharm
# ======================================================================================================================
# 利用 朴素bayes分类器 为文本分类
import numpy as np
import re
import matplotlib.pyplot as plt


def load_dataset():
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量对应每一个样本，1代表侮辱性词汇，0代表不是
    classvec = [0, 1, 0, 1, 0, 1]
    # 返回实验样本切分的词条、类别标签向量
    return postingList, classvec


# 去重创建一个包含所有出现词汇的list
def create_vocablist(dataset):
    gvocablist = set([])
    for sample in dataset:
        gvocablist = gvocablist | set(sample)
    return list(gvocablist)  # 集合向量化


# 样本转换成向量
def word2vec(sample, vocablist):
    returnvec = [0] * len(vocablist)  # 跟词特征向量同大小的向量
    for word in sample:
        if word in vocablist:
            returnvec[vocablist.index(word)] += 1
            # returnvec[vocablist.index(word)] = 1
    return returnvec


# 训练朴素贝叶斯分类器
# 基于二分类,二取值问题
def trainNB(Xdata, Ylabel):
    m, n = Xdata.shape
    # 先验概率
    ccnt1 = Ylabel[Ylabel[:, 0] == 1, :].shape[0]
    ccnt0 = Ylabel[Ylabel[:, 0] == 0, :].shape[0]
    pc1 = np.log((ccnt1 + 1) / (m + 2))
    pc0 = np.log((ccnt0 + 1) / (m + 2))
    # 条件计数
    wcnt1 = np.ones((1, n))
    wcnt0 = np.ones((1, n))

    for i in range(m):  # 遍历所有样本
        if Ylabel[i] == 1:
            wcnt1 += Xdata[i]
        else:
            wcnt0 += Xdata[i]
    # 贝叶斯概率
    pw1 = np.log(wcnt1 / ccnt1 + 2)
    pw0 = np.log(wcnt0 / ccnt0 + 2)
    return pc1, pc0, pw1, pw0


def classifyNB(samplevec, pc1, pc0, pw1, pw0):
    p1 = np.sum(samplevec * pw1) + pc1
    p0 = np.sum(samplevec * pw0) + pc0
    if p1 > p0:
        return 1
    else:
        return 0


def testNB():
    dataset, classvec = load_dataset()
    vocablist = create_vocablist(dataset)
    # 构建训练向量
    datatrain = []
    for sample in dataset:
        vec = word2vec(sample, vocablist)
        datatrain.append(vec)
    Xdata = np.array(datatrain)
    Ylabel = np.array(classvec).reshape(len(classvec), 1)
    pc1, pc0, pw1, pw0 = trainNB(Xdata, Ylabel)
    # 测试
    testentry = ["love", "my", "dalmation"]
    testvec = np.array(word2vec(testentry, vocablist))
    print("分类%d" % (classifyNB(testvec, pc1, pc0, pw1, pw0)))

    testentry = ["stupid", "garbage"]
    testvec = np.array(word2vec(testentry, vocablist))
    print("分类%d" % (classifyNB(testvec, pc1, pc0, pw1, pw0)))


# 简单切分文本,文本处理
def text_parse(inputstring):
    listoftokens = re.split(r"\W*", inputstring)
    return [tok.lower() for tok in listoftokens if len(tok) > 2]


def testSpam():
    # 读入数据
    doclist, classlist, fulltext = [], [], []
    for i in range(1, 26):  # 文件号1~25
        # 垃圾文件
        wordlist = text_parse(open("email/spam/%d.txt" % i).read())  # 提取词向量
        doclist.append(wordlist)  # 文档列表
        fulltext.extend(wordlist)  # 所有词汇量的集合
        classlist.append(1)
        # 正常文件
        wordlist = text_parse(open("email/ham/%d.txt" % i).read())
        doclist.append(wordlist)  # 文档列表
        fulltext.extend(wordlist)  # 所有词汇量的集合
        classlist.append(0)
    # 词特征向量
    vocablist = create_vocablist(doclist)
    # 构建数据
    trainingset = list(range(50))
    testset = []
    for i in range(10):
        randindex = np.int(np.random.uniform(0, len(trainingset)))
        testset.append(trainingset[randindex])
        del (trainingset[randindex])

    trainmat, trainclasses = [], []
    for docindex in trainingset:
        trainmat.append(word2vec(doclist[docindex], vocablist))
        trainclasses.append(classlist[docindex])
    # 训练
    pc1, pc0, pw1, pw0 = trainNB(np.array(trainmat), np.array(trainclasses).reshape(len(trainclasses), 1))
    # 精确度
    errorcount = 0
    for docindex in testset:
        wordvec = word2vec(doclist[docindex],vocablist)
        if classifyNB(np.array(wordvec),pc1,pc0,pw1,pw0)!=classlist[docindex]:
            errorcount += 1
            print(doclist[docindex])
    print("ERROR",np.float(errorcount)/len(testset))


if __name__ == "__main__":
    testNB()
    testSpam()
