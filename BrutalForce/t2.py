
import numpy as np


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization






def main():
    # 1.初始化数据
    data_dict = {-1: np.array([[1, 8],
                               [2, 3],
                               [3, 6]]),
                 1: np.array([[1, -2],
                              [3, -4],
                              [3, 0]])
                 }
    print("data_dict:\n", data_dict, ", type:", type(data_dict))

    for i in data_dict:
        print("i:", i)
        for xi in data_dict[i]:
            print("xi:", xi, ", type:", type(xi))

    # 2.初始化对象：构造svm，并调用相关函数
    svm = SupportVectorMachine()

    print("End!")


if __name__ == "__main__":
    main()






