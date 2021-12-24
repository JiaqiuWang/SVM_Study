"""
描述：支持向量机之SMO算法：按照标准版的算法进行修改
作者：王佳秋
日期：2021年12月24日
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


class SMOStruct:
    """按照John.platt的论文构造SMO的数据结构"""
    def __init__(self, X, y, C, kernel, alphas, b, errors, user_linear_optim):
        self.X = X  # 训练样本
        self.y = y  # 类别：label
        self.C = C  # regularization parameter 正则化常量，用于调整（过）拟合的程度
        self.kernel = kernel  # kernel function 核函数，实现了两个核函数：线性与高斯(RBF)核函数
        self.alphas = alphas  # lagrange multiplier拉格朗日乘子，与样本一一相对
        self.b = b  # scalar bias term 标量，偏移量
        self.errors = errors  # error cache 输出值误差：差值矩阵，用于存储判别函数实际值与预测值的差值，与样本一一相对
        # 存储的目的为递归、优化使得样本能够快速迭代，找到alpha值
        self.m, self.n = np.shape(self.X)  # store size(m) of training set and the
        # number of features(n) for each example训练 样本的个数和每个样本的features数量
        self.user_linear_optim = user_linear_optim  # Boolean值，判断模型是否使用线性核函数
        self.w = np.zeros(self.n)  # 初始化权重w的值，用于线性核函数


def plot_decision_boundary(model, ax, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
    """
    给出分割平面及支持平面，用的是等高线方法，怀疑绘制的分割平面和支持平面的准确性；
    generate coordinate grid of shape [resolution x resolution]
    and evaluate the model over the entire space
    :param model:
    :param ax:
    :param resolution: 坐标刻度的份数
    :param colors:
    :param levels:
    :return:
    """
    # X.[:, 0]取所有行的第0个数据，形成一个一维数组
    xrange = np.linspace(model.X[:, 0].min(), model.X[:, 0].max(), resolution)  # 横坐标最小值、最大值、分成100份
    yrange = np.linspace(model.X[:, 1].min(), model.X[:, 1].max(), resolution)
    grid = [[decision_function(model.alphas, model.y, model.kernel, model.X, np.array([xr, yr]), model.b)
             for xr in xrange] for yr in yrange]
    grid = np.array(grid).reshape(len(xrange), len(yrange))

    # Plot decision contours using grid and make a scatter plot of training data.
    ax.contour(xrange, yrange, grid, levels=levels, linewidth=(1, 1, 1), linestyles=('--', '-', '--'),
               colors=colors)  # 绘制坐标与等高线
    ax.scatter(model.X[:, 0], model.X[:, 1], c=model.y, cmap=cm.get_cmap('viridis'), lw=0, alpha=0.25)
    # 绘制所有样本的点

    # Plot support vectors (non-zero alphas) as circled points (linewidth > 0)
    mask = np.round(model.alphas, decimals=2) != 0.0  # 标记alphas中不等于0的点
    ax.scatter(model.X[mask, 0], model.X[mask, 1], c=model.y[mask], cmap=cm.get_cmap('viridis'),
               lw=1, edgecolors='k')
    return grid, ax


def linear_kernel(x, y, b=1):
    """线性核函数"""
    # returns the linear combination of arrays 'x' and 'y' with the optional
    # bias term 'b' (set to 1 by default).
    result = x @ y.T + b
    # print("x：\n", x, ", type:", type(x))
    # print("y：\n", x, ", type:", type(y))
    # print("x @ y.T:\n", x @ y.T, ", type:", type(x @ y.T), ", shape:", (x @ y.T).shape)
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
    """判别函数1：用于单一样本，主要用于get_error()"""
    if model.user_linear_optim:
        # 线性决策函数 Equation (J1)
        # return float(np.dot(model.w.T, model.X[i]) - model.b)
        return float(model.w.T @ model.X[i]) - model.b
    else:
        # 非线性决策函数 Equation (J10)
        return np.sum([model.alphas[j] * model.y[j] * model.kernel(model.X[j], model.X[i])
                       for j in range(model.m)]) - model.b


def decision_function(alphas, target, kernel, X_train, x_test, b):
    """判别函数2：用于多个样本，主要用于绘图
    Applies the SVM decision function to the input feature vectors in 'x_test'. """
    result = (alphas * target)
    # print("多样本决策函数中的kernel:", kernel)
    # result = kernel(X_train, x_test)
    # print("result:", result)
    result = (alphas * target) @ kernel(X_train, x_test) - b  # * . @ 两个Operators的区别
    # print("result:\n", result, ", shape:", result.shape)
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
        print("f(alpha{0})预测函数值{1}".format(sub, decision_function_output(model, sub)))
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
    print('Take_step步骤...')
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
    print("L:{0}, H:{1}".format(L, H))
    if L == H:
        return 0, model

    # 根据公式J16计算et, et=2k12-k11-k22, 分别计算样本1,2的核函数组合，目的在于计算eta
    # 也就是求一姐导数后的值，目的在于求alpha2 new
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    # 计算eta, equation J15: eta = K(x1, x1) + K(x2, x2) - 2K(x1, x2)
    eta = k11 + k22 - 2 * k12
    print("eta:", eta)

    # 如论文中所述，eta分两种情况：eta值为正positive，还是负数或0来计算alpha2 new
    # 第一种情况：eta值大于0
    if eta > 0:
        # equation J16 计算alpha2 new
        a2 = alpha2 + y2 * (E1 - E2) / eta
        # clip a2 based on bounds L & H，把a2夹到限定区间 equation J17
        if L < a2 < H:
            a2 = a2
        elif a2 <= L:
            a2 = L
        elif a2 >= H:
            a2 = H
        # print('old alpha2:', alpha2, ', new alpha2:', a2, ', 夹new alpha2到L,H限定区间：', a2)
    else:  # 如果eta值为负数或0，即<=0时，move new a2 to bound with greater objective function value
        # Equation J19，在特殊情况下，eta可能不为正not positive
        f1 = y1 * (E1 + model.b) - alpha1 * k11 - s * alpha2 * k12
        f2 = y2 * (E2 + model.b) - s * alpha1 * k12 - alpha2 * k22
        L1 = alpha1 + s * (alpha2 - L)
        H1 = alpha1 + s * (alpha2 - H)
        Lobj = L1 * f1 + L * f2 + 0.5 * (L1**2) * k11 + 0.5 * (L**2) * k22 + s * L * L1 * k12
        Hobj = H1 * f1 + H * f2 + 0.5 * (H1**2) * k11 + 0.5 * (H**2) * k22 + s * H * H1 * k12
        if Lobj < Hobj - eps:
            a2 = L
        elif Lobj > Hobj + eps:
            a2 = H
        else:
            a2 = alpha2
        print('old alpha2:', alpha2, ', new alpha2:', a2)

    # 当new a2 (alpha2) 千万分之一接近C或0时，就让他等于C或0
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C
    # 超过容差仍不能优化时，跳过
    # if examples can't be optimized within epsilon(eps), skip this pair
    if np.abs(a2 - alpha2) < eps * (a2 + alpha2 + eps):
        print("Within epsilon, skip this pair.")
        return 0, model

    # 根据a2 new计算a1 new, Equation J18: a1_new = a1_old + y1*y2*(a2_old - a2_new)
    # a1 = alpha1 + y1 * y2 * (alpha2 - a2)
    a1 = alpha1 + s * (alpha2 - a2)
    print('old alpha1:{0}, new alpha1:{1}'.format(alpha1, a1))

    # 更新 bias b(截距b)的值Equation J20: b1_new=E1+y1*k11(a1_new-a1_old)+y2*k21(a2_new-a2_old)+b_old
    b1 = E1 + y1 * k11 * (a1 - alpha1) + y2 * k12 * (a2 - alpha2) + model.b
    # Equation J21: b2_new=E2+y1(a1_new-a1_old)k12+y2(a2_new-a2_old)k22+b
    b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + model.b
    print('b1:', b1, ', b2:', b2)
    # Set new threshold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 < C:
        b_new = b1
    elif 0 < a2 < C:
        b_new = b2
    else:  # Average thresholds if both are bound
        b_new = (b1 + b2) * 0.5
    # print('b_new:', b_new)
    model.b = b_new
    # 原本在这里在更新模型中的b值，又由于更新误差的时候要计算新旧b值差，所以放到后面再更新


    # 当所训练模型为线性核函数时，根据Equation J22计算w的值w_new=w_old+y1(a1_new-a1_old)x1+y2(a2_new-a2_old)x2
    if model.user_linear_optim:
        print("old w:", model.w)
        model.w = model.w + y1 * (a1 - alpha1) * model.X[i1] + y2 * (a2 - alpha2) * model.X[i2]
        print("new w:", model.w)
    # 在alphas矩阵中分别更新a1, a2的值
    # Update model object with new alphas & threshold
    model.alphas[i1] = a1
    model.alphas[i2] = a2

    # 优化完了，更新差值矩阵的对应值。同时更新差值矩阵其他值
    model.errors[i1] = 0
    model.errors[i2] = 0
    # 更新差值 Equation 12 Ei_new=所有支持向量集合中每个样本j的误差=y_j*alpha_j*k_ij+b_new-y_i
    for k in range(model.m):  # 循环所有样本的数量
        if 0 < model.alphas[k] < model.C:
            print("old b:", model.b, ', new b:', b_new)
            print("alpha[{0}]:{1}, error[{0}]:{2}".format(k, model.alphas[k], model.errors[k]))
            model.errors[k] += y1 * (a1 - alpha1) * model.kernel(model.X[i1], model.X[k]) + \
                               y2 * (a2 - alpha2) * model.kernel(model.X[i2], model.X[k]) + \
                               model.b - b_new
            print('new error[{0}]:{1}'.format(k, model.errors[k]))
    # update model threshold，计算完model.b - b_new的值，在更新模型中的b值。

    print("更新完线性w值、alpha矩阵后返回new model.\n")
    return 1, model


def examine_example(i2, model):
    """
    寻找alpha1
    :param i2: 0~999，其实是下标
    :param model: model
    :return:
    """
    alpha2 = model.alphas[i2]  # 这里是old alpha2
    y2 = model.y[i2]  # old alpha2
    E2 = get_error(model, i2)  # error2=w{T}*x + b - y2
    r2 = E2 * y2

    # 重点：这一段的重点在于确定alpha1，也就是old alpha1,并送到take_step去analytically优化
    # 下面条件之一满足，进入if开始找第二个alpha，送到take_step进行优化
    # 如果满足第一个if语句说明不满足KTT约束条件，需要优化样本
    # 条件意思：在容差之内或alpha2需要优化的话，就开始优化。不需要优化满足KTT条件退出优化。
    print("alpha2的下标i2:", i2, ', alpha2:', alpha2, ', y2:', y2, ", E2:", E2)
    if (r2 < -tol and alpha2 < model.C) or (r2 > tol and alpha2 > 0):  # 违反KTT条件
        print('该i2:{}违反KTT条件，需要被优化'.format(i2))
        # 由于第一次的alpha矩阵都为0，所以没有不为0与C的alpha值，第一个if优化不被执行
        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            # print("不为0与C的alpha个数为：", len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]),
            #       ", 不为0与C的alpha数组下标：", np.where(model.alphas[(model.alphas != 0) & (model.alphas != model.C)])[0])
            # 先找那些不在0，C的点。选择Ei矩阵中差值做大的先进行优化
            # 要想|E2-E1|最大，只需在E2为正，选择最小的Ei作为E1
            # 在E2为负数时，选择最大的Ei作为E1

            if model.errors[i2] > 0:
                # print("model.errors:", model.errors)
                i1 = np.argmin(model.errors)  # 注：回沿轴的最小值的索引，而不是具体数值
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)

            step_result, model = take_step(i1, i2, model)
            if step_result:
                print("step_result为真！")
                return 1, model
            else:
                print('i1为空！')

        # 循环所有非0、非C的alpha值进行优化，随机选择起点
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):  # 从不为O与C的alpha数组中，随机选择一个i1，然后循环
            print("B-i1:{0}, i2:{1}".format(i1, i2))
            step_result, model = take_step(i1, i2, model)
            if step_result:
                print("step_result为真！")
                return 1, model

        # 由于初始alphas矩阵为0矩阵，第一次执行此处：当alpha2确定的时候，如何选择alpha1，循环所有的(m-1)alpha，随机选择起始点
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            print("C-i1:{0}, i2:{1}".format(i1, i2))
            # i1是alpha1的下标
            step_result, model = take_step(i1, i2, model)  # (3, 0, model)
            if step_result:
                print("step_result为真！")
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
    while (numChanged > 0) or examineAll:
        numChanged = 0
        if loopnum == 2000:
            break
        loopnum += 1

        if examineAll:
            loopnum1 += 1  # 记录顺序，一个一个选择alpha 的循环次数
            # 从0,1,2,3,...,m顺序选择a2的，送给examine_example选择alpha1，总共m(m-1)种选法
            for i in range(model.alphas.shape[0]):  # i 从0循环到999, model.alphas.shape为(1000,)
                examine_result, model = examine_example(i, model)  # 优化成功返回的examine_result为1，否则为0
                numChanged += examine_result

        else:  # 上面if里m(m-1)执行完的后执行
            loopnum2 += 1
            # loop over examples where alphas are not already at their limits
            # print("loopnum2中alpha不为0,C的个数：",
            #       len(np.where((model.alphas != 0) & (model.alphas != model.C))[0]), ', 数组：',
            #       np.where((model.alphas != 0) & (model.alphas != model.C))[0])
            # sys.exit()
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1

    print("loopNum:{0}, loopNum1:{1}, loopNum2:{2}".format(loopnum, loopnum1, loopnum2))
    return model



# make_blob需要解释一下
print("1 Python Main Function")
print(" 1.1生产训练样本X[x1, x2, ...xn]; Y=[y1, y2, ..., yn], yi=+-1")
# 1.1生产训练样本X[x1, x2, ...xn]; Y=[y1, y2, ..., yn], yi=+-1"
X_train, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
print(" 1.2数据预处理：数据预处理，使得经过处理的数据X符合正态分布，即均值为0，标准差为1，Y为-1或1")
# 1.2数据预处理：数据预处理，使得经过处理的数据X符合正态分布，即均值为0，标准差为1，Y为-1或1
scaler = StandardScaler()  # StandardScaler()以及fit_transform()函数的作用需要解释一下
# 训练样本异常大或异常小会影响样本的正确训练，如果数据的分部很分散也会影响
X_train_scaled = scaler.fit_transform(X_train, y)
y[y == 0] = -1

# 2.设置模型参数与对应的初始值：set model parameter and initial values
print('2 设置模型参数与对应的初始值')
C = 20.0  # 正则化超参，目标函数的约束   s.t. 0<=alpha<=C
m = len(X_train_scaled)  # 训练样本的数量
initial_alphas = np.zeros(m)  # 模型参数，每个样本对应一个alpha值，大多数样本的alpha值为0
# print("initial_alphas:", initial_alphas, ", shape:", initial_alphas.shape, ", type:", type(initial_alphas))
# 只有在support hyperplane之间的为C，之外的为0，在线之上为0<alpha<C
initial_b = 0.0  # 截距

# set tolerances  容差
tol = 0.01  # error tolerance 差值EI的容差。输出误差值=f(xi)-yi的值
eps = 0.01  # alpha tolerance 参数alpha误差值=alpha_new - alpha_old
# print(" 1.1 Set model parameters and initial values...")

# Instantiate model
print('3 实例化数据模型与参数初始值。')
model = SMOStruct(X=X_train_scaled, y=y, C=C, kernel=linear_kernel,
                  alphas=initial_alphas, b=initial_b, errors=np.zeros(m),
                  user_linear_optim=True)
print("4 计算所有样本的误差值=决策函数-真实值. ", model)
# Instantiate 差值矩阵
initial_error = decision_function(alphas=model.alphas, target=model.y, kernel=model.kernel,
                                  X_train=model.X, x_test=model.X, b=model.b) - model.y
model.errors = initial_error
# print(" 1.3 Set model parameters and initial values...")
# print("Initial model.errors:\n", model.errors)
np.random.seed(0)

print("5 开始训练... / Starting to fit...")
# 开始训练
output = fit(model)
# 绘制训练完，找到分割平面的图
fig, ax = plt.subplots()
grid, ax = plot_decision_boundary(output, ax)
plt.show()
print("End!")



