"""
描述：高斯核函数Gaussian Kernel用法详解
作者：王佳秋
日期：2021年11月12日
https://www.cxybb.com/article/jasneik/108150217
https://blog.csdn.net/hqh131360239/article/details/79061535
"""

import numpy as np


x3 = np.array([[0.41318279,  0.61920597], [1, 2]])
print(x3.ndim)

# print("x1 - x2:", x1 - x2)

def create_gauss_kernel(kernel_size=3, sigma=1, k=1):
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    X = np.linspace(-k, k, kernel_size)
    print("X:\n", X)
    Y = np.linspace(-k, k, kernel_size)
    print("Y:\n", Y)
    x, y = np.meshgrid(X, Y)
    x0 = 0
    y0 = 0
    gauss = 1 / (2 * np.pi * sigma**2) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return gauss


print("gauss_kernel:\n", create_gauss_kernel())

x4 = np.array([[0.3, 0.21], [0.8, 0.6]])
# print("x4 - x1:\n", x4 - x1)
# print("x1 - x4:\n", x1 - x4)

x5 = np.array([[0.3, 0.21, 0.4], [0.8, 0.6, 0.2]])
x6 = np.array([
    [[0.4, 0.6, 0.5], [0.6, 0.8, 0.1], [0.1, 0.5, 0.7]],
    [[0.3, 0.2, 0.8], [0.1, 0.4, 0.6], [1.2, 1.3, 0.8]]
              ])


def gausssion_kernel(x, y, sigma=1):
    # np.ndim()计算数组的维度，np.linalg.norm(x)求x向量平方和的平方根

    if np.ndim(x) == 1 and np.ndim(y) == 1:
        print("1wei ")
        result = np.exp(-(np.linalg.norm(x - y, ord=2))**2 / (2 * sigma**2))
        return result
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(-(np.linalg.norm(x - y, ord=2, axis=1)**2) / (2 * sigma**2))
        return result
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        print("多位*多位")
        result = np.exp(-(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], ord=2, axis=2)**2) / (2 * sigma**2))

print("x5[:, new]", x5[:, np.newaxis])
print(x6[np.newaxis, :])
#
print(x5[:, np.newaxis] - x6[np.newaxis, :])
gausssion_kernel(x5, x6)

a = np.array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]])
b=np.array([[1.],
       [1.],
       [1.],
       [1.]])
print("a-b:\n", a -b)

x3 = np.array([[0.41318279,  0.61920597], [1, 2]])
x1 = np.array([-0.48763336, -1.19492087])
x2 = np.array([0.41318279,  0.61920597])
print("x1,x2高斯", gausssion_kernel(x1, x2))
print("x1,x3高斯", gausssion_kernel(x1, x3))

print(1e-8.__round__())
