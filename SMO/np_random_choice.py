"""
描述：numpy.random.choice()用法详解
作者：王佳秋
日期：2021年11月11日
URL:https://blog.csdn.net/qq_25436597/article/details/79815800
"""

import numpy as np
import math

"""1.从大小为3的np.arange(5)生成一个均匀的随机样本："""
x = np.random.choice(5, 3)
print("x:", x)
x1 = np.random.randint(0, 5, 3)
print("x1:", x1)

"""2.从大小为3的np.arange(5)生成一个非均匀的随机样本："""
x3 = np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
print("x3:\n", x3)

"""3.从大小为3的np.arange(5)生成一个均匀的随机样本，没有替换（重复）：
True时，采样的元素会有重复；当replace指定为False时，采样不会重复。
"""
x4 = np.random.choice(5, 3, replace=False)
print("x4:", x4)
print("permutation:", np.random.permutation(np.arange(5))[:3])

"""4.从大小为3的np.arange(5)生成一个非均匀的随机样本，没有替换（重复）"""
x5 = np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0], replace=False)
print('x5:', x5)

"""5.上面例子中第一个参数都可以用一个任意的数组来代替，而不仅仅是整数。"""
x = ['pooh', 'rabbit', 'piglet', 'Christopher']
x6 = np.random.choice(x, 5, p=[0.5, 0.1, 0.1, 0.3])
print("x6:", x6)
