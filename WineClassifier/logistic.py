import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def random_grad_desc(X, y, rate, times):
    X = np.mat(X)
    m, n = np.shape(X)
    weight = np.ones((n, 1))   # N x 1 向量
    for i in range(times):
        for j in range(m):
            h = sigmoid(X[j] @ weight)   # 计算内积
            error = h - y[j]
            weight = weight - rate * X[j].transpose() * error
    return weight.T[0]


def loadDataSet():
    # 读取数据集
    dataSet = pd.read_csv('wine.csv')
    dataSet.insert(1, 'ones', 1)

    # dataSet = dataSet.sample(frac=1)   # 打乱重排
    dataSet = dataSet.dropna()         # 清除空行
    print(dataSet.tail())

    # 划分训练集和测试集
    dataNum = len(dataSet)
    k = int(dataNum * 0.7)
    trainData = dataSet[0: k]
    testData = dataSet[k:]
    x_test = testData.values[:, 1:]
    y_test = testData.values[:, 0]

    # 训练集信息
    x_train = trainData.values[:, 1:]
    y_train = trainData.values[:, 0]
    featureNum = len(x_train.T) - 1
    classType = []
    for c in dataSet.values[:, 0]:
        if c not in classType:
            classType.append(c)
    classNum = len(classType)
    print(dataNum, featureNum, classType, classNum)  # 178 // 2条样本，13项特征，3分类：1，2，3

    return dataSet, x_train, y_train, featureNum, classNum, classType, x_test, y_test


def main():
    # 导入数据集
    dataSet, x_train, y_train, featureNum, classNum, classType, x_test, y_test = loadDataSet()

    trainData_1 = dataSet.values[0:40, :]
    trainData_2 = dataSet.values[60:90, :]
    trainData_3 = dataSet.values[140:160, :]

    # 建立多个分类器
    weight = np.ones((classNum, featureNum + 1))

    x_clfn = trainData_1[:, 1:]
    y_clfn = trainData_1[:, 0]
    weight[0] = random_grad_desc(x_clfn, y_clfn, 0.01, 500)

    x_clfn = trainData_2[:, 1:]
    y_clfn = trainData_2[:, 0]
    weight[1] = random_grad_desc(x_clfn, y_clfn, 0.01, 500)

    x_clfn = trainData_3[:, 1:]
    y_clfn = trainData_3[:, 0]
    weight[2] = random_grad_desc(x_clfn, y_clfn, 0.01, 500)

    # 测试
    testData = np.vstack([dataSet.values[40:60, :], dataSet.values[90:140, :]])
    testData = np.vstack([testData, dataSet.values[160:177, :]])
    X = testData[:, 1:]
    y = testData[:, 0]
    p = [0, 0, 0]
    for i in range(len(X)):
        p[0] = sigmoid(weight[0] @ X[i])
        p[1] = 1 / (1 + np.exp(weight[0] @ X[i]) + np.exp(weight[1] @ X[i]))
        p[2] = 1 / (1 + np.exp(weight[0] @ X[i]) + np.exp(weight[1] @ X[i]) + np.exp(weight[1] @ X[i]))
        print(p, y[i])


if __name__ == '__main__':
    main()
