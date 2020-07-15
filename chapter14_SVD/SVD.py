import numpy as np
import matplotlib.pyplot as plt


def load_data():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


if __name__ == "__main__":
    # U, Sigma, VT = np.linalg.svd([[1, 1], [7, 7]])  # Sigma实际是一个矩阵,但是返回的是一个行向量(节省空间)
    # print("U", U, "Sigma", Sigma, "VT", VT)
    data = load_data()
    U, Sigma, VT = np.linalg.svd(data)
    print(Sigma)
    print(Sigma.shape)
    Sig3 = np.mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])  # 硬编码: 保留前三个奇异值构建对称矩阵
    print(U[:, :3] * Sig3 * VT[:3, :])  # U前三列 * Sig3 * VT前三行
