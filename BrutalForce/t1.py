import time

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')  # 图形显示风格的设置


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}  # red or blue
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # training
    def fit(self, data):
        self.data = data
        # key: value pair
        # {||w||: [w, b]}
        opt_dict = {}

        # 穷举x的正半轴到x的负半轴的所有直线的w和b值，180度转一圈。定义一个矩阵
        # cy = ax + b => y = (a/c)x + b, tan(theTa) = a/c = sin/cos
        reMatrix = lambda theta: np.array([np.sin(theta), np.cos(theta)])
        thetaStep = np.pi / 10
        # 自从生成数组
        transforms = [np.array(reMatrix(theta))
                      for theta in np.arange(0, np.pi, thetaStep)]

        # 将数据集拉平装到一个list当中，方便处理
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        # 找到数据集中的最大值和最小值
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # 定义步长
        step_size = [self.max_feature_value * 0.1, self.max_feature_value * 0.01,
                     self.max_feature_value * 0.001]

        # 寻找b的准备工作
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10
        for step in step_size:
            w = np.array([latest_optimum, latest_optimum])  # [80, 80]
            optimized = False

            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):  # arange(-1 * (8*5), 8*5, 0.08*5)
                    for transformation in transforms:
                        w_t = w * transformation  # [80, 80] * [0, 1], [80, 80] * [0.3, 0.95]
                        found_option = True

                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                else:
                    w = w-step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] * step * 2

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarage = (self.min_feature_value, self.max_feature_value)

        hyp_x_min = datarage[0]
        hyp_x_max = datarage[1]

        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 8],
                          [2, 3],
                          [3, 6]]),
             1: np.array([[1, -2],
                          [3, -4],
                          [3, 0]])
             }
print("data_dict:\n", data_dict)
for i in data_dict:
    print("i:", i)
    for xi in data_dict[i]:
        print("xi:", xi)

# 构造svm，并调用相关函数
svm = SupportVectorMachine()
svm.fit(data=data_dict)


# 预测
predict_us = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8],
              [-1, 0], [-2, 1]]
print("w:", svm.w, ", type:", type(svm.w), ". \nb:", svm.b)
for p in predict_us:
    svm.predict(p)

svm.visualize()







# 测试lambda函数
print("测试lambda函数")
f = lambda a, b, c, d: a * b * c * d
print(f(1, 2, 3, 4))

g = [lambda a: a * 2, lambda b: b * 3]
print(g[0](5))  # 调用
print(g[1](6))

add = lambda x, y: x + y
print("用法1：变量调用lambda：", add(2, 4))
time.sleep = lambda x: None
print("将lambda函数赋值给其他函数作为替换：", time.sleep(5))

print("Python内置函数接受函数作为参数")
print("内置函数(1)filter:", list(filter(lambda x: x % 3 == 0, [1, 2, 3])))
print("内置函数(2)sorted:", sorted([1, 2, 3, 4, 5, 6, 7, 8, 9], key=lambda x: abs(5 - x)))
print("内置函数(3)map:", list(map(lambda x: x+1, [1, 2, 3])))
# print("内置函数(4)reduce:", reduce(lambda a, b: '{}, {}'.format(a, b)), [1, 2, 3, 4, 5, 6, 7, 8, 9])
# https://zhuanlan.zhihu.com/p/58579207  lambda函数的用法


reMatrix = lambda theta: np.array([np.sin(theta), np.cos(theta)])
thetaStep = np.pi / 10
print("thetaStep:", thetaStep)
# 自从生成数组
transforms = [np.array(reMatrix(theta)) for theta in np.arange(0, np.pi, thetaStep)]

reMatrix = lambda theta: np.array([np.sin(theta), np.cos(theta)])
for theta in np.arange(0, np.pi, thetaStep):
    print("theta:", theta)
    print("reMatrix(theta):", reMatrix(theta))
    print("np.array:", np.array(reMatrix(theta)))
print("transforms:", transforms)

dataArr = [-0.2, -1.1, 0, 2.3, 4.5, 0.0]
print("输入的数据：", dataArr)
print("使用sign()求输入数据的符号：", np.sign(dataArr))

