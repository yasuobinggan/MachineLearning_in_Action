# @Time : 2019/7/7 8:56 
# @Author : HXT
# @File : test.py 
# @Software: PyCharm
# ======================================================================================================================
import numpy as np

# # a = numpy.mat([[1,2,3],[4,5,6]])
# #
# # def change(z):
# #     z[0,1]=1000
# #
# # change(a)
# #
# # print(a)
# #
# # x,y = a.shape
# # print(x,y)
#
# # 注意矩阵相乘结果为一阶阵的维度，和原始一阶阵的维度
# # a = np.eye(6)
# # b = np.arange(1,31).reshape(15,2)
# # print(np.dot(a,b))
# # print(a)
# # print(a[:,0])
# # print(a[0,-1])
# # print(False)
# # par = {}
# # print(len(par))
#
# # array = np.ones((6,6))
# # array[:,2] += 1
# # array[0,2] = 3
# # array[3,2] = 3
# # print(array)
# # print(a[array[:,2]==1,:])
#
# # flag = set()
# # flag.add(1)
# # flag.add(2)
# # flag = np.array(flag)
# # print(flag)
# # print(a[flag,:])
#
# def f():
#     global b
#     b = 10
#
# f.a = 1
# print(f.a)
# a = 0
# # f()
# # print(type(type(b)))
# # print(type(type(b).__name__))
# print(a)
# print(f.a)
# f.a = 99
# print(f.a)


# list1 = ["H"]
# def delete(inputlist):
#     del inputlist[0]
# delete(list1)
# print(list1)

# a = np.arange(1,25,1).reshape(4,6)
# # print(a)
# # print(np.sum(a))
# # print(np.sum(a,axis=0))
#
# y = np.ones((4,1))
# y[:,:] = a[:,-1].reshape(4,1)
# print(y)
#
#
# print(a[:,:].shape)
# print(a[:,0].shape)
# print(a[2,:].shape)
# print(a[:,0:1].shape)
# print(a[:,0:2].shape)


# a = np.zeros((1, 4)).reshape(4, 1)
# b = np.arange(5,11)
# print(b.shape )
# print(b,b.T)
# print(a.T)


# a = np.arange(1,2)
# b = np.arange(3,4)
# c = np.arange(5,6)
# print((a*b*c).shape)


# flag1 = False
# flag2 = True
# a = 0
# if flag1:
#     a = 1
# elif flag2:
#     a = 2
#
# print(a)
#
# print(len(range(1)))
# print(len(range(50)))


zd = {1:100,2:200,3:300}
print(zd.get(1,0))
print(zd.get(2,200))
print(zd.get(5,0))

print(11/3)