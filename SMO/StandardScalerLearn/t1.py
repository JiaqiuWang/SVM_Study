

"""
URL;https://www.javaxks.com/?p=30038#transform%E5%87%BD%E6%95%B0%E7%9A%84%E7%AE%80%E4%BB%8B
https://zhuanlan.zhihu.com/p/97389097
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_blobs



# 标准化
trainX, trainY = make_blobs(n_samples=50, centers=3, n_features=3, random_state=1)
print(trainX, trainY, ", type:", type(trainX))
scaler = StandardScaler()
print(scaler.fit(trainX, trainY))
StandardScaler(copy=True, with_mean=True, with_std=True)
print("mean of trainX:", scaler.mean_, ", standard variation:", scaler.var_,
      "scale:", scaler.scale_, ", number:", scaler.n_samples_seen_)

# 归一化
scaler_mmc = MinMaxScaler()
print(scaler_mmc.fit(trainX, trainY))
print("Maximum:", scaler_mmc.data_max_)
print("Minimum:", scaler_mmc.data_min_)
print("range of data:", scaler_mmc.data_range_)

"""
项目的数据集一般都会分为 训练集和测试集，训练集用来训练模型，测试集用来验证模型效果。
注意了，是用训练集进行拟合，然后对训练集、测试集都用拟合好的”模型“进行转换，一定要明白这个逻辑！！
不能对训练集和测试集都使用 fit_transform，虽然这样对测试集也能正常转换（归一化或标准化），
但是两个结果不是在同一个标准下的，具有明显差异。
"""
testX, testY = make_blobs(n_samples=50, centers=3, n_features=3, random_state=0)
# 训练集操作
# newTrainX = scaler.fit_transform(trainX, trainY)
# 测试集操作
scaler2 = StandardScaler()
newTrainX = scaler2.fit_transform(trainX, trainY)
newTestX = scaler2.transform(testX)
print("newTestX:\n", newTestX)

newTestX = scaler2.fit_transform(testX)
print("newTestX':\n", newTestX)  # 这个是错误的







