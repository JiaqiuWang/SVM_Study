"""
描述：支持向量机之SMO算法
作者：王佳秋
日期：2021年10月19日
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


class SMOStruct:
    """按照John.platt的论文构造SMO的数据结构"""

    def __init__(self, X, y, C, kernel, alpha, b, errors, user_linear_optim):
        self.X = X  # 训练样本
        self.y = y  # 类别：label
        self.C = C  # regularization parameter 正则化常量，用于调整（过）拟合的程度
        self.kernel = kernel  # kernel function 核函数，实现了两个核函数：线性与高斯(RBF)核函数
        self.alphas = alpha  # lagrange multiplier拉格朗日乘子，与样本一一相对
        self.b = b  # scalar bias term 标量，偏移量
        self.errors = errors  # error cache 差值矩阵，用于存储判别函数实际值与预测值的差值，与样本一一相对
        # 存储的目的为递归、优化使得样本能够快速迭代，找到alpha值
        self.m, self.n = np.shape(self.X)  # store size(m) of training set and the
        # number of features(n) for each example
        self.user_linear_optim = user_linear_optim  # 判断模型是否使用线性核函数
        self.w = np.zeros(self.n)  # 初始化权重w的值，主要用于线性核函数


def linear_kernel(x, y, b=1):
    """线性核函数"""
    # returns the linear combination of arrays 'x' and 'y' with the optional
    # bias term 'b' (set to 1 by default).
    result = x @ y.T + b
    return result  # Note the @ operator for matrix multiplications


def main():
    # make_blob需要解释一下
    print("1.Python Main Function")
    # 生产测试数据，训练样本
    X_train, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
    # StandardScaler()以及fit_transfrom函数的作用需要解释一下
    scaler = StandardScaler()  # 数据预处理，使得经过处理的数据符合正态分布，即均值为0，标准差为1
    # 训练样本异常大或异常小会影响样本的正确训练，如果数据的分部很分散也会影响
    X_train_scaled = scaler.fit_transform(X_train, y)
    y[y == 0] = -1

    # set model parameter and initial values
    C = 20.0  # 正则化超参，目标函数的约束   s.t. 0<alpha<C
    m = len(X_train_scaled)  # 训练样本的数量
    initial_alphas = np.zeros(m)  # 模型参数，每个样本对应一个alpha值，大多数样本的alpha值为0
    # 只有在support hyperplane之间的为C，之外的为0.
    initial_b = 0.0  # 截距

    # set tolerances  容差
    tol = 0.01  # error tolerance 差值EI的容差
    eps = 0.01  # alpha tolerance

    # Instantiate model
    model = SMOStruct(X_train_scaled, y, C, linear_kernel, initial_alphas, initial_b,
                      np.zeros(m), user_linear_optim=True)

    print("End!")


if __name__ == "__main__":
    main()
