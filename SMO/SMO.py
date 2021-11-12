"""
描述：支持向量机之SMO算法
作者：王佳秋
日期：2021年10月19日
45:51
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


class SMOStruct:
    """按照John.platt的论文构造SMO的数据结构"""

    def __init__(self, X, y, C, kernel, alpha, b, errors, user_linear_optim, tol, eps):
        self.X = X  # 训练样本
        self.y = y  # 类别：label
        self.C = C  # regularization parameter 正则化常量，用于调整（过）拟合的程度
        self.kernel = kernel  # kernel function 核函数，实现了两个核函数：线性与高斯(RBF)核函数
        self.alphas = alpha  # lagrange multiplier拉格朗日乘子，与样本一一相对
        self.b = b  # scalar bias term 标量，偏移量
        self.errors = errors  # error cache 输出值误差：差值矩阵，用于存储判别函数实际值与预测值的差值，与样本一一相对
        # 存储的目的为递归、优化使得样本能够快速迭代，找到alpha值
        self.m, self.n = np.shape(self.X)  # store size(m) of training set and the
        # number of features(n) for each example训练 样本的个数和每个样本的features数量
        self.user_linear_optim = user_linear_optim  # Boolean值，判断模型是否使用线性核函数
        self.w = np.zeros(self.n)  # 初始化权重w的值，用于线性核函数
        self.tol = tol  # 函数值的容差预测值-真实值
        self.eps = eps  # alpha tolerance 参数alpha误差值=alpha_new - alpha_old


def linear_kernel(x, y, b=1):
    """线性核函数"""
    # returns the linear combination of arrays 'x' and 'y' with the optional
    # bias term 'b' (set to 1 by default).
    result = x @ y.T + b
    print("x：\n", x, ", type:", type(x))
    print("y：\n", x, ", type:", type(y))
    print("x @ y.T:\n", x @ y.T, ", type:", type(x @ y.T), ", shape:", (x @ y.T).shape)
    # print("x @ y:\n", x @ y.T, ", dot():", np.dot(x, y))
    return result  # Note the @ operator for matrix multiplications


def gaussian_kernel(x, y, sigma=1):
    """
    高斯核函数
    np.ndim()计算数组的维度，np.linalg.norm(x)求x向量平方和的平方根
    :param x:
    :param y:
    :param sigma:
    :return: the gaussian similarity of arrays 'x' and 'y' with the kernel width parameter 'sigma'
    set to 1 by default.
    """
    result = None
    if np.ndim(x) == 1 and np.ndim(y) == 1:  # 一个数字
        result = np.exp(-(np.linalg.norm(x - y, ord=2))**2 / (2 * sigma**2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):  # 高维数组
        result = np.exp(-(np.linalg.norm(x - y, ord=2, axis=1)**2) / (2 * sigma**2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(-(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], ord=2, axis=2)**2) / (2 * sigma**2))
    return result


def decision_function_output(model, i):
    """判别函数1：用于单一样本"""
    if model.user_linear_optim:
        # 线性决策函数 Equation (J1)
        # return float(np.dot(model.w.T, model.X[i]) - model.b)
        return float(model.w.T @ model.X[i]) - model.b
    else:
        # 非线性决策函数 Equation (J10)
        return np.sum([model.alphas[j] * model.y[j] * model.kernel(model.X[j], model.X[i])
                       for j in range(model.m)]) - model.b


def decision_function(alphas, target, kernel, X_train, x_test, b):
    """判别函数2：用于多个样本
    Applies the SVM decision function to the input feature vectors in 'x_test'. """
    # result = (alphas * target)
    print("kernel:", kernel)
    result = kernel(X_train, x_test)
    # print("result:", result)
    result = (alphas * target) @ kernel(X_train, x_test) - b  # * . @ 两个Operators的区别
    print("result:\n", result, ", shape:", result.shape)
    return result


def get_error(model, sub):
    """
    按角标获取样本误差
    :param sub:
    :param model:
    :return:
    """
    if 0 < model.alphas[sub] < model.C:  # 在此区间的alpha属支持向量，落在超平面上，是要找的alphas值，
        # 直接从计算完放到误差矩阵中的值即可
        return model.errors[sub]
    else:  # alpha=0或C, =0时，样本点在超平面外表示该样本分类正确，=C时样本点在超平面间表示分类错误。
        return decision_function_output(model, sub) - model.y[sub]


# 选择了alpha2，alpha1后开始第一步优化，然后迭代，“第二层循环、内循环”
# 主要的优化步骤在这里发生
def take_step(i1, i2, model):
    """
    :param i1: alpha1的角标
    :param i2: alpha2的角标
    :param model:
    :return:
    """
    # skip if chosen alphas are the same.由于i1,i2都是下标，则i1==i2的时候说明匹配到同一个alpha，则跳过
    if i1 == i2:
        return 0, model
    # 如果i1 != i2的情况下，
    alpha1 = model.alphas[i1]  # old_alpha1
    alpha2 = model.alphas[i2]  # old_alpha2
    y1 = model.y[i1]  # old_y1
    y2 = model.y[i2]  # old_y2
    E1 = get_error(model, i1)  # old_E1
    E2 = get_error(model, i2)  # old_E2
    s = y1 * y2
    # 计算alpha的边界，L, H
    # Computing L & H, the bounds on new possible alpha values
    if y1 != y2:
        # y1, y2异号，使用Equation(J13)
        L = max(0, alpha2 - alpha1)
        H = min(model.C, model.C + alpha2 - alpha1)
    elif y1 == y2:
        # y1, y2同号，使用Equation(J14)
        L = max(0, alpha1 + alpha2 - model.C)
        H = min(model.C, alpha1 + alpha2)
    if L == H:
        return 0, model

    # 根据公式J16计算et, et=2k12-k11-k22, 分别计算样本1,2的核函数组合，目的在于计算eta
    # 也就是求一姐导数后的值，目的在于求alpha2 new
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])

    return


def examine_example(i2, model):
    """
    寻找alpha1
    :param i2: 0~999，其实是下标
    :param model: model
    :return:
    """
    alpha2 = model.alphas[i2]  # 这里是old alpha2
    y2 = model.y[i2]
    E2 = get_error(model, i2)  # error2=w{T}*x + b - y2
    r2 = E2 * y2

    # 重点：这一段的重点在于确定alpha1，也就是old alpha1,并送到take_step去analytically优化
    # 下面条件之一满足，进入if开始找第二个alpha，送到take_step进行优化
    # 如果满足第一个if语句说明不满足KTT约束条件，需要优化样本
    # 条件意思：在容差之内或alpha2需要优化的话，就开始优化。不需要优化满足KTT条件退出优化。
    if (r2 < -model.tol and alpha2 < model.C) or (r2 > model.tol and alpha2 > 0):  # 违反KTT条件
        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            # 先找那些不在0，C的点。选择Ei矩阵中差值做大的先进行优化
            # 要想|E2-E1|最大，只需在E2为正，选择最小的Ei作为E1
            # 在E2为负数时，选择最大的Ei作为E1
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)  # 注：回沿轴的最小值的索引，而不是具体数值
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

    # 循环所有非0、非C的alpha值进行优化，随机选择起点
    for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                      np.random.choice(np.arange(model.m))):
        step_result, model = take_step(i1, i2, model)
        if step_result:
            return 1, model


    # 由于初始alphas矩阵为0矩阵，第一次执行此处：当alpha2确定的时候，如何选择alpha1，循环所有的(m-1)alpha，随机选择起始点
    for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
        # i1是alpha1的下标
        step_result, model = take_step(i1, i2, model)  # (3, 0, model)
        if step_result:
            return 1, model

    # 先看上面的if语句，如果if条件不满足，说明KKT条件已满足，找其它样本进行优化，否则执行下面语句，退出
    return 0, model


def fit(model):
    """
    训练函数
    :param model:
    :return:
    """
    numChanged = 0  # 优化的、返回的结果，如果优化成功返回1，否则返回0。如果每次优化返回1，则是一个计数器功能。
    examineAll = 1  # 从0号元素开始优化，如果所有的元素优化完了，则把1置为0.

    # loop num record
    # 计数器，记录优化时的循环次数
    loopnum = 0
    loopnum1 = 0
    loopnum2 = 0

    # 当numChanged = 0 and examineAll = 0 时，循环退出。实际循环按顺序执行所有样本，也就是第一个if中的循环。
    # 并且else中for循环没有可优化的alpha，目标函数收敛了：在容差之内，并且满足KTT条件
    # 则循环退出，如果执行次数超过一定阈值时仍未收敛，也退出。
    # 重点：确定alpha2，也就是old alpha2或alpha2下标，old alpha2和old alpha1都是启发式选择。
    while numChanged > 0 or examineAll:
        numChanged = 0
        if loopnum == 2000:
            break
        loopnum += 1

        if examineAll:
            loopnum1 += 1  # 记录顺序，一个一个选择alpha 的循环次数
            # 从0,1,2,3,...,m顺序选择a2的，送给examine_example选择alpha1，总共m(m-1)种选法
            for i in range(model.alphas.shape[0]):
                examine_result, model = examine_example(i, model)  # 优化成功返回的examine_result为1，否则为0
                numChanged += examine_result
        else:  # 上面if里m(m-1)执行完的后执行
            loopnum2 += 1
            # loop over examples where alphas are not already at their limits
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                numChanged += examine_result

        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineALL = 1

    print("loopNum:{0}, loopNum1:{1}, loopNum2:{2}".format(loopnum1, loopnum1, loopnum2))
    return model


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
    C = 20.0  # 正则化超参，目标函数的约束   s.t. 0<=alpha<=C
    m = len(X_train_scaled)  # 训练样本的数量
    initial_alphas = np.zeros(m)  # 模型参数，每个样本对应一个alpha值，大多数样本的alpha值为0
    print("alphas.shape():", initial_alphas.shape)
    # 只有在support hyperplane之间的为C，之外的为0，在线之上为0<=alpha<=C
    initial_b = 0.0  # 截距

    # set tolerances  容差
    tol = 0.01  # error tolerance 差值EI的容差。输出误差值=f(xi)-yi的值
    eps = 0.01  # alpha tolerance 参数alpha误差值=alpha_new - alpha_old

    # Instantiate model
    model = SMOStruct(X=X_train_scaled, y=y, C=C, kernel=linear_kernel,
                      alpha=initial_alphas, b=initial_b, errors=np.zeros(m),
                      user_linear_optim=True, tol=tol, eps=eps)
    print(" 1.1 Model created...")
    # print("errors:", model.errors, ", shape:", model.errors.shape)
    # Instantiate 差值矩阵
    print("model.kernel:", model.kernel)
    initial_error = decision_function(alphas=model.alphas, target=model.y, kernel=model.kernel,
                                      X_train=model.X, x_test=model.X, b=model.b) - model.y
    print("初始化误差值矩阵initial_error:\n", initial_error)
    model.errors = initial_error
    np.random.seed(0)

    '''
    X_train, y = make_circles(n_samples=500, noise=0.2, factor=0.1, random_state=1)
    X_train, y = make_moons(n_samples=500, noise=0.2, random_state=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y)
    y[y == 0] = -1
    print("X_train:\n", X_train)
    print("y:\n", y)

    # set model parameters and initial values
    C = 1.0
    m = len(X_train_scaled)
    initial_alphas = np.zeros(m)
    initial_b = 0.0

    # set tolerance
    tol = 0.01  # error tolerance
    eps = 0.01  # alpha tolerance

    # instantiate model
    model = SMOStruct(X=X_train_scaled, y=y, C=C, lambda x, y: gaussian_kernel(x, y, sigma=0.5),
                      alpha=initial_alphas, b=initial_b, errors=np.zeros(m),
                      user_linear_optim=False)

    # initialize error cache
    # 先把这个注释掉
    initial_error = decision_function(model.alphas, model.y, model.kernel, model.X, model.X,
                                      model.b) - model.y
    initial_error = np.zeros(m)
    print('initial error:\n', initial_error)
    model.errors = initial_error
    '''

    print("Starting to fit...")
    # 开始训练
    output_model = fit(model)
    # 绘制训练完，找到分割平面的图
    # fig, ax = plt.subplots()
    # grid, ax = plot_decision_boundary(output, ax)
    print("End!")


if __name__ == "__main__":
    main()
