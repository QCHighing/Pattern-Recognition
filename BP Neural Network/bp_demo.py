import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使 plt 中的中文字体能够显现
plt.rcParams['axes.unicode_minus'] = False  # 使 plt 中的中文字体能够显现


# 定义 sigmoid 函数
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# 原始数据
# 人数（单位：万人）
sqrs = [20.55, 22.44, 25.37, 27.13, 29.45, 30.10, 30.96, 34.06, 36.42, 38.09, 39.13, 39.99, 41.93, 44.59, 47.30, 52.89, 55.73, 56.76, 59.17, 60.63]
# 机动车数(单位：万辆)
sqjdcs = [0.6, 0.75, 0.85, 0.9, 1.05, 1.35, 1.45, 1.6, 1.7, 1.85, 2.15, 2.2, 2.25, 2.35, 2.5, 2.6, 2.7, 2.85, 2.95, 3.1]
# 公路面积(单位：万平方公里)
sqglmj = [0.09, 0.11, 0.11, 0.14, 0.20, 0.23, 0.23, 0.32, 0.32, 0.34, 0.36, 0.36, 0.38, 0.49, 0.56, 0.59, 0.59, 0.67, 0.69, 0.79]
# 公路客运量(单位：万人)
glkyl = [5126, 6217, 7730, 9145, 10460, 11387, 12353, 15750, 18304, 19836, 21024, 19490, 20433, 22598, 25107, 33442, 36836, 40548, 42927, 43462]
# 公路货运量(单位：万吨)
glhyl = [1237, 1379, 1385, 1399, 1663, 1714, 1834, 4322, 8132, 8936, 11099, 11203, 10524, 11115, 13320, 16762, 18673, 20724, 20803, 21804]

# 输入数据
x_data = np.array([sqrs, sqjdcs, sqglmj], np.float64)
x_train = x_data[:, 0:15]  # 前 15 个数据作为训练集
x_test = x_data[:, 15:20]  # 后 5 个数据作为测试集

# 真实值
t_data = np.array([glkyl, glhyl], np.float64)
t_train = t_data[:, 0:15]
t_test = t_data[:, 15:20]

# 归一化数据（0,1 标准化）
minmax_x_train = MinMaxScaler()
x_train_std = minmax_x_train.fit_transform(x_train)
minmax_t_train = MinMaxScaler()
t_train_std = minmax_t_train.fit_transform(t_train)

# 构建网络
input_size = 3
hidden_size = 8
output_size = 2  # 超参数
iters_num = 1000  # 迭代次数
train_size = x_train.shape[1]
learning_rate = 0.035  # 学习率

# 初始化权值和阈值
weight_init_std = 0.01
W1 = weight_init_std * np.random.randn(hidden_size, input_size)
b1 = np.zeros([hidden_size, 1])
W2 = weight_init_std * np.random.randn(output_size, hidden_size)
b2 = np.zeros([output_size, 1])

for i in range(iters_num):
    # 正向传播
    z1 = np.dot(W1, x_train_std) + b1
    h = sigmoid(z1)
    z2 = np.dot(W2, h) + b2
    y = z2
    # 计算梯度
    delta2 = t_train_std - y
    dW2 = np.dot(delta2, np.transpose(h))
    db2 = np.dot(delta2, np.ones([train_size, 1]))
    delta1 = np.dot(np.transpose(W2), delta2) * h * (1 - h)
    dW1 = np.dot(delta1, np.transpose(x_train_std))
    db1 = np.dot(delta1, np.ones([train_size, 1]))
    # 参数更新
    W2 += learning_rate * dW2
    b2 += learning_rate * db2
    W1 += learning_rate * dW1
    b1 += learning_rate * db1

# 训练集结果
z1 = np.dot(W1, x_train_std) + b1
h = sigmoid(z1)
z2 = np.dot(W2, h) + b2
y_train = z2
originoutput_train = minmax_t_train.inverse_transform(y_train)  # 还原网络归一化结果
newk_train = originoutput_train[0, :]
newh_train = originoutput_train[1, :]

# 测试集结果
minmax_x_test = MinMaxScaler()
minmax_t_test = MinMaxScaler()
x_test_std = minmax_x_test.fit_transform(x_test)
t_test_std = minmax_t_test.fit_transform(t_test)
z1 = np.dot(W1, x_test_std) + b1
h = sigmoid(z1)
z2 = np.dot(W2, h) + b2
y_test = z2
originoutput_test = minmax_t_test.inverse_transform(y_test)
newk_test = originoutput_test[0, :]
newh_test = originoutput_test[1, :]

# 结果绘图
plt.figure()
time = np.linspace(1990, 2009, 20)
time1 = np.linspace(1990, 2004, 15)  # 训练数据时间
time2 = np.linspace(2005, 2009, 5)  # 测试数据时间
# 绘值公路客运量对比图；
plt.subplot(2, 1, 1)
plt.title(u'神经网络客运量与货运量训练和测试对比图')
plt.xticks(np.arange(1990, 2010, 2))
plt.plot(time, glkyl, 'b--+', label=u'实际客运量')
plt.plot(time1, newk_train, 'r--o', label=u'训练网络输出客运量')
plt.plot(time2, newk_test, 'g--o', label=u'测试网络输出客运量')
plt.legend(loc='upper left')
plt.ylabel(u'客运量/万人')
# 绘制公路货运量对比图；
plt.subplot(2, 1, 2)
plt.xticks(np.arange(1990, 2010, 2))
plt.plot(time, glhyl, 'b--+', label=u'实际货运量')
plt.plot(time1, newh_train, 'r--o', label=u'训练输出货运量')
plt.plot(time2, newh_test, 'g--o', label=u'测试网络输出货运量')
plt.legend(loc='upper left')
plt.xlabel(u'年份')
plt.ylabel(u'货运量/万吨')
plt.show()
