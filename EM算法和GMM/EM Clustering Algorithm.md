

```python
import numpy as np
import pandas as pd
import scipy.stats as st             # 计算多元高斯概率分布函数值
import matplotlib.pyplot as plt
from sklearn import preprocessing    # 归一化处理
```

# 1. 数据处理

## 1.1 读取数据


```python
dataset = pd.read_csv('./sample_dataset.csv')     # pandas.core.frame.DataFrame 类型
X = dataset.values[:, 0:-1]
```

## 1.2 绘制原始分类示图


```python
A = dataset.values
A1 = A[A[:, 2] == 0]
A2 = A[A[:, 2] == 1]
A3 = A[A[:, 2] == 2]
plt.figure('GMM')
plt.plot(A1[:, 0], A1[:, 1], 'rs', label="class1")
plt.plot(A2[:, 0], A2[:, 1], 'bo', label="class2")
plt.plot(A3[:, 0], A3[:, 1], 'g+', label="class3")
plt.legend(loc="best")
plt.title("GMM")
```

[![i5q2ZT.png](https://s1.ax1x.com/2018/11/04/i5q2ZT.png)](https://imgchr.com/i/i5q2ZT)


## 1.3 归一化（可选）


```python
# X_scale = preprocessing.scale(X)    # 归一化
X_scale = X                           # 不进行归一化
```


```python
# 手动归一化
# x1 = (x1 - min(x1)) / (max(x1) - min(x1))
# x2 = (x2 - min(x2)) / (max(x2) - min(x2))
```

# 2. 估计GMM参数

## 2.1 设定参数


```python
K = 3             # 高斯分量个数，即类别 K=3
N, D = X.shape    # 数据量 N, 特征维数 D
```

## 2.2 设定迭代初值


```python
mu = np.random.rand(K, D)          # 均值矩阵：K行D维向量
sigma = np.ones((K, D))            # 方差矩阵：K行D维向量
alpha = np.array([1.0 / K] * K)    # 组合系数：K个元素，和是1
print(f'\n初始参数：\nmu=\n{mu}\n\nsigma=\n{sigma}\n\nalpha=\n{alpha}\n')
```


    初始参数：
    mu=
    [[0.93168761 0.25647956]
     [0.81838383 0.04418795]
     [0.65451381 0.16478288]]

    sigma=
    [[1. 1.]
     [1. 1.]
     [1. 1.]]

    alpha=
    [0.33333333 0.33333333 0.33333333]



## 2.3 迭代


```python
times = 200
gamma = np.zeros((N, K))
gamma_num = np.zeros((N, K))     # 响应度分子部分
gamma_den = np.zeros((N, K))     # 响应度分母部分
gamma_sum = np.zeros((1, K))      # 各分量响应度之和
for i in range(times):
    # E step:依据当前分量模型的参数，计算各模型对所有数据的响应度 gamma
    for k in range(K):
        # 计算协方差矩阵
        cov = sigma[k] * np.eye(D)
        normal = st.multivariate_normal(mean=mu[k], cov=cov)  # 构造多元高斯模型，可通过pdf方法计算phi值
        gamma_num[:, k] = normal.pdf(X_scale) * alpha[k]
    gamma_den = np.array([np.sum(gamma_num, 1)] * K).T
    gamma = gamma_num / gamma_den
    gamma_sum = sum(gamma)
    # M step:更新各分量模型的参数
    for k in range(K):
        for d in range(D):
            mu[k, d] = X_scale[:, d] @ gamma[:, k] / gamma_sum[k]
            sigma[k, d] = ((X_scale[:, d] - mu[k, d])**2) @ gamma[:, k] / gamma_sum[k]
        alpha[k] = gamma_sum[k] / N
    if (i + 1) % 100 == 0:
        print(f'\n第{i+1}次迭代：\nmu=\n{mu}\n\nsigma=\n{sigma}\n\nalpha=\n{alpha}\n')
```


    第100次迭代：
    mu=
    [[ 5.93880138e+00  1.99259544e+00]
     [ 3.01917592e+00  1.98328428e+00]
     [-4.24005257e-03  2.02822367e+00]]
    sigma=
    [[0.36986907 0.29061748]
     [0.2866718  0.21467908]
     [0.1881047  0.10686893]]
    alpha=
    [0.19434035 0.48045827 0.32520138]


    第200次迭代：
    mu=
    [[ 5.93880138e+00  1.99259544e+00]
     [ 3.01917592e+00  1.98328428e+00]
     [-4.24005257e-03  2.02822367e+00]]
    sigma=
    [[0.36986907 0.29061748]
     [0.2866718  0.21467908]
     [0.1881047  0.10686893]]
    alpha=
    [0.19434035 0.48045827 0.32520138]



# 3. 聚类

## 3.1 根据响应度聚类


```python
clsindex = gamma.argmax(axis=1)  # 根据响应度聚类
category = np.argsort(mu[:, 0])  # 根据中心位置的x1坐标调整类别次序
class1 = np.array([X[i] for i in range(N) if category[clsindex[i]] == 0])
class2 = np.array([X[i] for i in range(N) if category[clsindex[i]] == 1])
class3 = np.array([X[i] for i in range(N) if category[clsindex[i]] == 2])
```

## 3.2 绘制聚类示图


```python
plt.figure('GMM Clustering')
plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
plt.plot(class3[:, 0], class3[:, 1], 'g+', label="class3")
plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.show()
```

[![i5qWoF.png](https://s1.ax1x.com/2018/11/04/i5qWoF.png)](https://imgchr.com/i/i5qWoF)

## 3.3 准确率


```python
y = dataset.values[:, -1].reshape(N, 1)
csum = 0
for i in range(N):
    if y[i] == category[clsindex[i]]:
        csum += 1
accuracy = csum / N; accuracy
```




    0.992


