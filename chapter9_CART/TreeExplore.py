# @Time : 2019/8/4 21:38 
# @Author : HXT
# @File : TreeExplore.py 
# @Software: PyCharm
# ======================================================================================================================
import numpy as np
import tkinter
import chapter9_CART.ModelTree as MT
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# 画图测试
# root = tkinter.Tk()
# mylabel = tkinter.Label(root, text="Hello World")
# mylabel.grid()
# root.mainloop()


# 画图
def re_draw(tolS, tolN):
    re_draw.f.clf()
    re_draw.a = re_draw.f.add_subplot(111)
    Yhyp = None
    if chkBtnVar.get():  # 模型树
        if tolN < 2:
            tolN = 2
        mytree = MT.create_tree(re_draw.rawDat, MT.modelleaf, MT.modelerr, (tolS, tolN))
        Yhyp = MT.create_forecast(mytree, re_draw.testDat, MT.modeltree_eval)
    else:  # 回归树
        mytree = MT.create_tree(re_draw.rawDat, ops=(tolS, tolN))
        Yhyp = MT.create_forecast(mytree, re_draw.testDat)

    re_draw.a.scatter(np.array(re_draw.rawDat[:, 0]), np.array(re_draw.rawDat[:, 1]), s=5)
    re_draw.a.plot(re_draw.testDat, Yhyp, linewidth=2.0)
    re_draw.canvas.draw()


# 获得输入框的值
def getInputs():
    try:
        tolN = int(tolNentry.get())  # 强制转换
    except:
        tolN = 10
        print("Enter Integer for tolN")
        tolNentry.delete(0, tkinter.END)
        tolNentry.insert(0, "10")
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("Enter Float for tolS")
        tolSentry.delete(0, tkinter.END)
        tolSentry.insert(0, "1.0")
    return tolN, tolS


def draw_newtree():
    tolN, tolS = getInputs()
    re_draw(tolS, tolN)


root = tkinter.Tk()
tkinter.Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)

tkinter.Label(root, text="tolN").grid(row=1, column=0)
tolNentry = tkinter.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, "10")  # 预设初始值

tkinter.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = tkinter.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, "1.0")  # 预设初始值

# 画树按钮
tkinter.Button(root, text="ReDraw", command=draw_newtree).grid(row=1, column=2, rowspan=3)

# 是否选择模型树
chkBtnVar = tkinter.IntVar()  # 设定一个初始值
chkBtn = tkinter.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

# 函数属性(全局变量)
re_draw.rawDat = np.mat(MT.load_dataset("sine.txt"))  # 数据集
m, n = re_draw.rawDat.shape

re_draw.testDat = np.arange(min(re_draw.rawDat[:, 0]), max(re_draw.rawDat[:, 0]), 0.01)  # 测试集 #TODO
print(re_draw.testDat.shape)
# print(re_draw.testDat)

# TODO
re_draw.f = Figure(figsize=(5, 4), dpi=100)
re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
re_draw.canvas.draw()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

re_draw(1.0, 10)

root.mainloop()
