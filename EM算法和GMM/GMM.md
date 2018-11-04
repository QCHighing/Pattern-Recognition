

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

# 1. 设定GMM参数


```python
K = 3           # 高斯分量个数
N = 1000        # 数据量
```

#### 为便于绘图示意，这里选用二元高斯作为分量，参数如下：


```python
mu = np.array([[0, 2], [3, 2], [6, 2]])                     # 均值矩阵
sigma = np.array([[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]])      # 方差矩阵
alpha = (0.3, 0.5, 0.2)                                     # 组合系数矩阵，和必须是1
```

# 2. 随机生成数据（二选一）

## 2.1 按照比例生成数据（顺序）


```python
beta = np.array([sum(alpha[0:i]) for i in range(K+1)])
k = np.uint(beta * N);k
gmm = np.zeros((N, 3))
for i in range(K):
    cov = sigma[i] * np.eye(2)
    size = k[i+1] - k[i]
    gmm[k[i]:k[i+1], 0:2] = np.random.multivariate_normal(mean=mu[i], cov=cov, size=size)
    gmm[k[i]:k[i+1], 2] = i
```

## 2.2 随机选择分量生成（乱序）


```python
beta = np.array([sum(alpha[0:i+1]) for i in range(K)])
gmm = np.zeros((N, 3))
for i in range(N):
    rdnum = np.random.rand()
    index = np.argwhere(beta > rdnum)[0, 0]            # 轮盘赌选法
    cov = sigma[index] * np.eye(2)
    gmm[i, 0:2] = np.random.multivariate_normal(mean=mu[index], cov=cov)
    gmm[i, 2] = index
```

## 2.3 统计各分量的数据量


```python
tmp = gmm[:, 2].tolist()
for i in range(K):
    print(tmp.count(i))
```

    324
    484
    192


# 3. 写入csv文件


```python
df = pd.DataFrame({'x1': gmm[:, 0], 'x2': gmm[:, 1], 'class': gmm[:, 2]})
df.to_csv('./sample_dataset.csv', index=False, sep=',')
```

# 4. 读出csv文件


```python
dataset = pd.read_csv('sample_dataset.csv')
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.046013</td>
      <td>1.867824</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.103495</td>
      <td>1.933776</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.647689</td>
      <td>2.169746</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.821279</td>
      <td>1.859590</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.852580</td>
      <td>2.054247</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# 5. 绘制数据示图


```python
A = dataset.values
A1 = A[A[:, 2] == 0]
A2 = A[A[:, 2] == 1]
A3 = A[A[:, 2] == 2]
plt.plot(A1[:, 0], A1[:, 1], "bo")
plt.plot(A2[:, 0], A2[:, 1], "rs")
plt.plot(A3[:, 0], A3[:, 1], "g+")
plt.show()
```

![i57hid.png](https://s1.ax1x.com/2018/11/04/i57hid.png)

