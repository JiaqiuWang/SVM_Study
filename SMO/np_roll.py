"""
描述：numpy.roll()用法详解
作者：王佳秋
日期：2021年11月11日
URL:https://blog.csdn.net/lxq1997/article/details/83543709
"""

import numpy as np
import math


x = np.arange(10)
print("x:", x)

x1 = np.roll(x, 2)  # axis为None，则会先进行扁平化，然后再向水平滚动2个位置
print("x1:", x1)

x = np.reshape(x, (2, 5))
print("x:\n", x)

x2 = np.roll(x, 1)  # axis为None，则会先进行扁平化，然后再向水平滚动1个位置
print("x2:\n", x2)

x3 = np.roll(x, 1, axis=0)  # 5列同时向垂直方向滚动1个位置
print("x3:\n", x3)

x4 = np.roll(x, 1, axis=1)  # 2行同事向水平方向移动1个位置
print("x4:\n", x4)

x5 = np.array([0, 1, 441])
print('测试x5:', np.roll(x5, 836))
for i in np.roll(x5, 836):
    print("i:", i)
