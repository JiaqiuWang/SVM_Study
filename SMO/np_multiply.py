"""
描述：numpy中线性代数详解SMO算法中的函数值的矩阵乘法
作者：王佳秋
日期：2021年11月19日
URL:https://wizardforcel.gitbooks.io/pyda-2e/content/4.html
"""

import numpy as np
import math


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print("x:", x, ", y:", y)
print("x.dot(y):\n", x.dot(y))
# x.dot(y)等价于np.dot(x, y)
print("np.dot(x, y):", np.dot(x, y))
# print("np.dot(y, y):\n", np.dot(y, y))  # 错误 不能3*2 dot 3*2 运算
z = np.ones(3)
print("z:", z, ", shape:", z.shape)
print(np.dot(x, z.T))
print('@:', x @ z)

def linear_kernel(x, y, b=1):
    """线性核函数"""
    result = x @ y.T + b
    return result

alpha = np.array([2, 3, 4])
y = np.array([1, -1, -1])
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
FX = alpha * y @ linear_kernel(X, X)
print("FX:\n", FX)
fx1 = alpha[0]*y[0]*linear_kernel(np.array([1, 2, 3]), np.array([1, 2, 3])) + \
      alpha[1]*y[1]*linear_kernel(np.array([4, 5, 6]), np.array([1, 2, 3])) + \
      alpha[2]*y[2]*linear_kernel(np.array([7, 8, 9]), np.array([1, 2, 3]))
print("fx1:", fx1)
















