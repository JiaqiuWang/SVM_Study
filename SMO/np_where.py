"""
描述：numpy.where()用法详解
作者：王佳秋
日期：2021年11月2日
"""

import numpy as np
import math

'''
第一种用法：np.where(condition, x, y): 满足条件condition，输出x，否则输出y
'''
aa = np.arange(10)
print("aa:{0}".format(aa))
ex1 = np.where(aa, 1, -1)  # 0为False，所以第一个输出-1
print("ex1:", ex1)
ex2 = np.where([[True, False], [True, True]], [[1, 2], [3, 4]], [[9, 8], [7, 6]])
print("ex2:\n", ex2)
a = 7
ex3 = np.where([[a > 5, a < 5], [a == 10, a == 7]],
               [['chosen', 'not chosen'], ['chosen', 'not chosen']],
               [['not chosen', 'chosen'], ['not chosen', 'chosen']])
print("ex3:\n", ex3)


'''
第二种用法：np.where(condition), 没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。
这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
'''
a = np.array([2, 4, 6, 8, 10])
ex1 = np.where(a > 5)
print("ex1:返回的是索引，不是元素", ex1)
ex2 = a[np.where(a > 5)]
ex3 = a[a > 5]
print("ex2:{0} and ex3:{1}".format(ex2, ex3))
ex4 = np.where([[0, 1], [1, 0]])  # 注意：输出的是坐标，不是元素[[0,1],[1,0]]的真值为两个1，各自的第一维坐标为[0,1]，第二维坐标为[1,0] 。
print("ex4:", ex4)
ex5 = np.arange(27).reshape(3, 3, 3)
print("ex5:", ex5)
ex6 = np.where(ex5 > 5)
print('ex6:\n', ex6)
ex7 = ex5[np.where(ex5 > 5)]
print(math.floor(len(ex7)/3))  # 向下取整
print("替换成坐标的元素：\n", np.array(ex7).reshape(math.floor(len(ex7)/3), 3))
