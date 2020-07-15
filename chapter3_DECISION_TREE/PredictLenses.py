# @Time : 2019/7/31 21:25 
# @Author : HXT
# @File : PredictLenses.py 
# @Software: PyCharm
# ======================================================================================================================
import chapter3_DECISION_TREE.DecisionTree as DT
import chapter3_DECISION_TREE.DecisionTreePloter as DP

fr = open("lenses.txt")
lenses = [sample.strip().split("\t") for sample in fr.readlines()]
lensesfeatures = ["age", "prescript", "astigmatic", "tearrate"]
lensestree = DT.create_tree(lenses, lensesfeatures)
print(lensesfeatures)
DP.createplot(lensestree)
