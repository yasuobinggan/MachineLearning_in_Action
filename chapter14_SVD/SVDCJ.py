#  @Time  : 2019/9/3 11:17
#  @Author : HXT
#  @Site  :
#  @File  : SVDCJ.py
#  @Software: PyCharm
# ======================================================================================================================
# 压缩图像
import numpy as np
import matplotlib.pyplot as plt


# 输出二进制的手写数字图像
def print_mat(inmat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inmat[i, k]) > thresh:
                print(1,end="")
            else:
                print(0,end="")
        print()


# 图像压缩
def img_compress(numsv=3, thresh=0.8):
    # 读入数据
    myl = []
    for line in open("0_5.txt").readlines():  # 按行读取
        newrow = []
        for i in range(32):
            newrow.append(int(line[i]))  # 每行按列读取
        myl.append(newrow)
    mymat = np.mat(myl)
    # 输出原始矩阵
    print("****original matrix******")
    print_mat(mymat, thresh)
    U, Sigma, VT = np.linalg.svd(mymat)  # 奇异值分解
    # 构建对角奇异值矩阵
    SigmaRecon = np.mat(np.zeros((numsv, numsv)))
    for k in range(numsv):
        SigmaRecon[k, k] = Sigma[k]
    # 利用降维后的矩阵还原矩阵
    reconmat = U[:, :numsv] * SigmaRecon * VT[:numsv, :]
    print("****reconstructed matrix using %d singular values******" % numsv)
    print_mat(reconmat, thresh)


if __name__ == "__main__":
    img_compress(numsv=2)
