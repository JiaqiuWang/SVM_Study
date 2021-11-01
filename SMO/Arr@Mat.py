"""
描述：Python中向量、矩阵学习
作者：王佳秋
日期：2021年10月29日
"""

import numpy as np


"""一.一维向量（行、列）：
在numpy中一维向量用一位数组array([1, 1, 1])表示，既能表示行向量也能表示列向量，一维向量转置后还是原来的样子（因为存储结构是数组）
"""
# 1.向量、矩阵表示：numpy的向量表示是通过array数组来实现的
v1 = np.array([1, 2, 8])
v1T1 = v1.transpose()  # 转置矩阵
v1T2 = v1.T  # 转置矩阵
print("向量\nv1:", v1, ", type:", type(v1))
print("v1.transpose():", v1T1)
print("v1.T:", v1T2)

# 2.矩阵
v1 = np.array([[1, 1, 1]])  #
print("矩阵\nv1:", v1, ", type:", type(v1))
# 两行三列的矩阵
v2 = np.array([[1, 1, 2], [1, 1, 0]])
print("矩阵\nv2:", v2, ", type:", type(v2))


"""二.向量计算：线性代数矩阵乘法"""
# 1. 线性代数矩阵乘法，行乘列，再相加
v1 = np.array([1, 0, 1])
v11 = np.array([1, 1, 1])
v2 = np.array([[1, 1, 2], [1, 1, 0]])
v22 = np.array([[1, 2, 3], [2, 1, 2]])
print("一维向量dot:", np.dot(v1, v11))
print("一维向量*二维向量：", np.dot(v2, v1))
print("@符号的计算：", v1 @ v11)

print("二维向量*二维向量：", np.dot(v2, v22.transpose()))
print("@符号的计算：", v2 @ v22.T)


"""三.对应位置相乘"""
print("一维向量multiply:", np.multiply(v1, v1))
print("一维向量*:", v1 * v11)
print("一维向量multiply二维向量：\n", np.multiply(v1, v2), ", type:", type(np.multiply(v1, v2)))


"""四.加减法"""
print("一维向量相加：", v1 + v11)
print("二维向量相加：", v1 + v2)
print("一维向量向量相减：", v1 - v11)
print("一维向量减二维向量：", v1 - v2)


"""五.矩阵的另一种表示mat
(1) 在创建矩阵的专用字符串中，矩阵的行与行之间用分号隔开，行内的元素之间用空格隔开。
(2) 用T属性获取转置矩阵
"""
# 1.创建矩阵：mat函数创建任意矩阵
a = np.mat('1 2 3; 4 5 6')
print("mat函数创建的任意矩阵：\n", a, ", type:", type(a))
aTp = a.transpose()
print("a的转置矩阵transpose：\n", aTp)
aT = a.T
print("a的转置矩阵transpose：\n", aT)
B = np.mat(np.arange(12).reshape(3, 4))
print("元素值为0-11的3*4维矩阵：\n", B)


"""六.构建全0矩阵"""
A = np.zeros((4, 5))
print("全0矩阵：\n", A)


"""七.构建单位矩阵"""
C = np.eye(3)
print("单位矩阵：\n", C)


"""八.构建符合矩阵"""
D = C * 2
print("单位矩阵*2：\n", D)
E = np.bmat('C D; D C')
print("连接bind符合矩阵：\n", E)


print("np.random.seed(0):", np.random.seed(0))
