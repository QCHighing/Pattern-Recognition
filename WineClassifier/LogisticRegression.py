import numpy as np
import pandas as pd


def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


def stocGradAscent1(dataMat, labelMat, numIter=150):
    m, n = np.shape(dataMat)
    alpha = 0.1
    weight = np.ones(n)  # float
    # weight = np.random.rand(n)
    for j in range(numIter):
        dataIndex = list(range(m))  # range 没有del 这个函数　　所以转成list  del 见本函数倒数第二行
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))  # random.uniform(0,5) 生成0-5之间的随机数
            # 生成随机的样本来更新权重。
            h = sigmoid(sum(dataMat[randIndex] * weight))
            error = labelMat[randIndex] - h
            weight = weight + alpha * error * np.array(dataMat[randIndex])  # !!!!一定要转成array才行
            # dataMat[randIndex] 原来是list  list *2 是在原来的基础上长度变为原来2倍，
            del(dataIndex[randIndex])  # 从随机list中删除这个
    return weight


def classifyVector(inX, weight):  # 输入测试带测试的向量 返回类别
    prob = sigmoid(sum(inX * weight))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest(weight, dataMat, labelMat):
    errorCount = 0
    numTestVec = 0.0
            for i in range(21):
                linearr.append(float(currtline[i]))  # 转换为float
            if int(classifyVector(np.array(linearr), trainWeights)) != int(currtline[21]):
                errorCount += 1  # 输入带分类的向量，输出类别，类别不对，errorCount ++
            errorRate = float(errorCount) / numTestVec
            print("the error rate of this test is : %f" % errorRate)
    return errorRate


def multiTest(weight, dataMat, labelMat):  # 所有测试集的错误率
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest(weight, dataMat, labelMat)
    print("after %d iterations the average error rate is : %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':

    # 读取数据集
    dataSet = pd.read_csv('wine.csv')

    # 划分训练集和测试集
    dataNum = len(dataSet)
    trainData = dataSet[0:dataNum // 2]
    testData = dataSet[dataNum // 2:]

    # 训练集信息
    x_train = trainData.values[:, 1:]
    y_train = trainData.values[:, 0]
    featureNum = len(x_train.T)
    classType = []
    for c in dataSet.values[:, 0]:
        if c not in classType:
            classType.append(c)
    classNum = len(classType)

    # 测试集
    x_test = testData.values[:, 1:]
    y_test = testData.values[:, 0]

    # weight = gradAscent(x_train, y_train)
    weight = stocGradAscent1(x_train, y_train)
    print(weight)
    multiTest(weight, x_test, y_test)  # 真实数据集上测试
