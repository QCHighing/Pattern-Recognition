import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stopCriterion(type, value, threshold):
    # 设定三种不同的停止策略
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


def model(X, theta):
    p = sigmoid(np.dot(X, theta))
    ret = 0 if p > 0.5 else 1
    return ret


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        grad[j] = np.sum(term) / len(X)
    return grad


def descent(X, y, theta, batchSize, stopType, thresh, alpha, n=100):
    # 梯度下降求解
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            # X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1
        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh):
            break
    return theta, i - 1, costs, grad, time.time() - init_time


# 梯度下降法
def grad_desc(data_mat, label_mat, rate, times):
    """
    :param data_mat: 数据特征
    :param label_mat: 数据标签
    :param rate: 速率
    :param times: 循环次数
    :return: 参数
    """
    data_mat = np.mat(data_mat)
    label_mat = np.mat(label_mat)
    m, n = np.shape(data_mat)
    weight = np.ones((n, 1))
    for i in range(times):
        h = sigmoid(data_mat * weight)
        error = h - label_mat
        weight = weight - rate * data_mat.transpose() * error
    return weight


# 随机梯度下降法
def random_grad_desc(data_mat, label_mat, rate, times):
    """
    :param data_mat: 数据特征
    :param label_mat: 数据标签
    :param rate: 速率
    :param times: 循环次数
    :return: 参数
    """
    data_mat = np.mat(data_mat)
    m, n = np.shape(data_mat)
    weight = np.ones((n, 1))
    for i in range(times):
        for j in range(m):
            h = sigmoid(data_mat[j] * weight)
            y = 1 if label_mat[j] == 1 else 0
            error = h - y
            weight = weight - rate * data_mat[j].transpose() * error
    return weight


def predict(X, theta):
    for x in X:
        yield model(x, theta)


def main():
    # 读取数据集
    dataSet = pd.read_csv('wine.csv')
    dataSet.insert(1, 'ones', 1)

    dataSet = dataSet.sample(frac=1)   # 打乱重排
    dataSet = dataSet.dropna()         # 清除空行
    print(dataSet.tail())

    # 划分训练集和测试集
    dataNum = len(dataSet)
    k = int(dataNum * 0.7)
    trainData = dataSet[0: k]
    testData = dataSet[k:]

    # 训练集信息
    x_train = trainData.values[:, 1:]
    y_train = trainData.values[:, 0]
    featureNum = len(x_train.T) - 1
    classType = []
    for c in dataSet.values[:, 0]:
        if c not in classType:
            classType.append(c)
    classNum = len(classType)
    print(dataNum // 2, featureNum, classType, classNum)  # 178 // 2条样本，13项特征，3分类：1，2，3

    # 数据归一化（可选）
    maxV, minV = x_train.max(axis=0), x_train.min(axis=0)
    x_train_norm = (x_train - minV) / maxV - minV
    # mean = x_train.mean(axis=0)     # 计算各个特征的均值、方差
    # std = x_train.std(axis=0)
    # x_train_norm = (x_train - mean) / std

    theta = random_grad_desc(x_train, y_train, rate=0.1, times=1000)
    print(theta)

    x_test = testData.values[:, 1:]
    y_test = trainData.values[:, 0]

    for reallabel, predlabel in zip(y_test, predict(x_test, theta)):
        flag = 'True' if reallabel == predlabel else 'Flase'
        print(predlabel, reallabel, flag)


if __name__ == '__main__':
    main()
