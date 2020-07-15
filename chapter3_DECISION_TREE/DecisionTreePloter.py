# coding=utf-8

# @Time : 2019/7/28 14:59
# @Author : HXT
# @File : DecisionTreePloter.py
# @Software: PyCharm
# ======================================================================================================================
# 使用matplotlib注解绘制决策树
import matplotlib.pyplot as plt

# 定义文本框和箭头
# 转换成字典形式
DecisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 非叶结点
LeafNode = dict(boxstyle="round4", fc="0.8")  # 叶结点
ArrowArgs = dict(arrowstyle="<-")  # 箭头


# print(decisionnode, leafnode, arrowargs)

# TODO plt语法不懂

# 预设两棵树
def retrieveTree(i):
    listoftrees = [{"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
                   {"no surfacing": {0: "no", 1: {"flippers": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}]
    return listoftrees[i]


# 决策树的叶结点数量
def get_numleaf(MyTree):
    numleafs = 0
    firstkey = next(iter(MyTree))
    child = MyTree[firstkey]
    for key in child.keys():
        if type(child[key]).__name__ == "dict":
            numleafs += get_numleaf(child[key])
        else:
            numleafs += 1
    return numleafs


# 决策树高(不包括叶结点)
def get_treedepth(MyTree):
    maxdepth = 0
    firstkey = next(iter(MyTree))
    child = MyTree[firstkey]
    for key in child.keys():
        if type(child[key]).__name__ == "dict":
            maxdepth = 1 + max(maxdepth, get_treedepth(child[key]))
        else:
            maxdepth = max(maxdepth, 1)
    # maxdepth += 1
    return maxdepth


# print(get_numleaf(mytree))
# print(get_treedepth(mytree))

# 画线上标记
def plot_midtext(CntrPt, ParentPt, TxtString):
    xmid = (ParentPt[0] - CntrPt[0]) / 2.0 + CntrPt[0]
    ymid = (ParentPt[1] - CntrPt[1]) / 2.0 + CntrPt[1]
    createplot.ax1.text(xmid, ymid, TxtString)


# 画点
def plot_node(NodeTxt, CenterPt, ParentPt, NodeType):
    createplot.ax1.annotate(NodeTxt, xy=ParentPt, xycoords="axes fraction", xytext=CenterPt,
                            textcoords="axes fraction", va="center", ha="center",
                            bbox=NodeType, arrowprops=ArrowArgs)


# 递归绘图
def plot_tree(MyTree, ParentPt, NodeTxt):
    numleafs = get_numleaf(MyTree)
    depth = get_treedepth(MyTree)
    firstkey = next(iter(MyTree))
    CntrPt = (plot_tree.xOff + (1.0 + float(numleafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)  # 当前结点位置

    plot_midtext(CntrPt, ParentPt, NodeTxt)
    plot_node(firstkey, CntrPt, ParentPt, DecisionNode)
    # 孩子结点
    seconddict = MyTree[firstkey]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD  #
    for key in seconddict.keys():
        # 非叶结点
        if type(seconddict[key]).__name__ == "dict":
            plot_tree(seconddict[key], CntrPt, str(key))
        # 叶结点
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(seconddict[key], (plot_tree.xOff, plot_tree.yOff), CntrPt, LeafNode)
            plot_midtext((plot_tree.xOff, plot_tree.yOff), CntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD  #


# 绘图总函数
def createplot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createplot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plot_tree.totalW = float(get_numleaf(inTree))
    plot_tree.totalD = float(get_treedepth(inTree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(inTree, (0.5, 1.0), "")

    # plot_node("DecisionNode", (0.5, 0.1), (0.1, 0.5), DecisionNode)
    # plot_node("LeafNode", (0.8, 0.1), (0.3, 0.8), LeafNode)

    plt.show()


mytree = retrieveTree(0)
# mytree["no surfacing"][3]="may be"
createplot(mytree)
