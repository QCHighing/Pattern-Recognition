

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

# 1. 导入数据集

## 1.1 读取文件 csv


```python
path = 'wine.csv'
dataset = pd.read_csv(path)
```

## 1.2 查看（首尾）数据


```python
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
      <th>class</th>
      <th>alcohol</th>
      <th>malic acid</th>
      <th>ash</th>
      <th>alcalinity of ash</th>
      <th>magnesium</th>
      <th>total phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid phenols</th>
      <th>proanthocyanins</th>
      <th>color intensity</th>
      <th>hue</th>
      <th>diluted wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.tail()
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
      <th>class</th>
      <th>alcohol</th>
      <th>malic acid</th>
      <th>ash</th>
      <th>alcalinity of ash</th>
      <th>magnesium</th>
      <th>total phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid phenols</th>
      <th>proanthocyanins</th>
      <th>color intensity</th>
      <th>hue</th>
      <th>diluted wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>172</th>
      <td>3</td>
      <td>14.16</td>
      <td>2.51</td>
      <td>2.48</td>
      <td>20.0</td>
      <td>91</td>
      <td>1.68</td>
      <td>0.70</td>
      <td>0.44</td>
      <td>1.24</td>
      <td>9.7</td>
      <td>0.62</td>
      <td>1.71</td>
      <td>660</td>
    </tr>
    <tr>
      <th>173</th>
      <td>3</td>
      <td>13.71</td>
      <td>5.65</td>
      <td>2.45</td>
      <td>20.5</td>
      <td>95</td>
      <td>1.68</td>
      <td>0.61</td>
      <td>0.52</td>
      <td>1.06</td>
      <td>7.7</td>
      <td>0.64</td>
      <td>1.74</td>
      <td>740</td>
    </tr>
    <tr>
      <th>174</th>
      <td>3</td>
      <td>13.40</td>
      <td>3.91</td>
      <td>2.48</td>
      <td>23.0</td>
      <td>102</td>
      <td>1.80</td>
      <td>0.75</td>
      <td>0.43</td>
      <td>1.41</td>
      <td>7.3</td>
      <td>0.70</td>
      <td>1.56</td>
      <td>750</td>
    </tr>
    <tr>
      <th>175</th>
      <td>3</td>
      <td>13.27</td>
      <td>4.28</td>
      <td>2.26</td>
      <td>20.0</td>
      <td>120</td>
      <td>1.59</td>
      <td>0.69</td>
      <td>0.43</td>
      <td>1.35</td>
      <td>10.2</td>
      <td>0.59</td>
      <td>1.56</td>
      <td>835</td>
    </tr>
    <tr>
      <th>176</th>
      <td>3</td>
      <td>13.17</td>
      <td>2.59</td>
      <td>2.37</td>
      <td>20.0</td>
      <td>120</td>
      <td>1.65</td>
      <td>0.68</td>
      <td>0.53</td>
      <td>1.46</td>
      <td>9.3</td>
      <td>0.60</td>
      <td>1.62</td>
      <td>840</td>
    </tr>
  </tbody>
</table>
</div>



## 1.3 数据统计信息


```python
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 177 entries, 0 to 176
    Data columns (total 14 columns):
    class                   177 non-null int64
    alcohol                 177 non-null float64
    malic acid              177 non-null float64
    ash                     177 non-null float64
    alcalinity of ash       177 non-null float64
    magnesium               177 non-null int64
    total phenols           177 non-null float64
    flavanoids              177 non-null float64
    nonflavanoid phenols    177 non-null float64
    proanthocyanins         177 non-null float64
    color intensity         177 non-null float64
    hue                     177 non-null float64
    diluted wines           177 non-null float64
    proline                 177 non-null int64
    dtypes: float64(11), int64(3)
    memory usage: 19.4 KB
    

**无空数据，不需要额外处理**


```python
dataset.describe()
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
      <th>class</th>
      <th>alcohol</th>
      <th>malic acid</th>
      <th>ash</th>
      <th>alcalinity of ash</th>
      <th>magnesium</th>
      <th>total phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid phenols</th>
      <th>proanthocyanins</th>
      <th>color intensity</th>
      <th>hue</th>
      <th>diluted wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
      <td>177.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.932203</td>
      <td>12.994237</td>
      <td>2.326384</td>
      <td>2.364407</td>
      <td>19.466667</td>
      <td>99.762712</td>
      <td>2.296497</td>
      <td>2.036441</td>
      <td>0.360734</td>
      <td>1.592260</td>
      <td>5.034689</td>
      <td>0.959412</td>
      <td>2.617401</td>
      <td>747.949153</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.773078</td>
      <td>0.809641</td>
      <td>1.112355</td>
      <td>0.273670</td>
      <td>3.327599</td>
      <td>14.320209</td>
      <td>0.627353</td>
      <td>0.997087</td>
      <td>0.123904</td>
      <td>0.573694</td>
      <td>2.303684</td>
      <td>0.227710</td>
      <td>0.707886</td>
      <td>315.484679</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>12.360000</td>
      <td>1.600000</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.740000</td>
      <td>1.220000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.210000</td>
      <td>0.790000</td>
      <td>1.960000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>13.050000</td>
      <td>1.860000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.360000</td>
      <td>2.140000</td>
      <td>0.340000</td>
      <td>1.560000</td>
      <td>4.680000</td>
      <td>0.970000</td>
      <td>2.780000</td>
      <td>675.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>13.670000</td>
      <td>3.030000</td>
      <td>2.550000</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.880000</td>
      <td>0.430000</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 1.4 按类划分


```python
class_1st = dataset[dataset['class'] == 1] 
class_2nd = dataset[dataset['class'] == 2]
class_3rd = dataset[dataset['class'] == 3]
```

## 1.5 划分训练集与测试集（各类3/7分）


```python
lens = np.array([len(class_1st), len(class_2nd), len(class_3rd)])
k = np.uint8(lens * 0.7)

class_1st_1 = class_1st[:k[0]]; class_1st_2 = class_1st[k[0]:]
class_2nd_1 = class_2nd[:k[1]]; class_2nd_2 = class_2nd[k[1]:]
class_3rd_1 = class_3rd[:k[2]]; class_3rd_2 = class_3rd[k[2]:]

dataset_1 = pd.concat([class_1st_1, class_2nd_1, class_3rd_1])
dataset_2 = pd.concat([class_1st_2, class_2nd_2, class_3rd_2])
```

# 2. 分类器设计

## 2.1 sigmoid 函数


```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

**计算时可能会发生溢出，但不用理会。比如：**


```python
np.exp(1000)
```

    d:\program files\python\lib\site-packages\ipykernel_launcher.py:1: RuntimeWarning: overflow encountered in exp
      """Entry point for launching an IPython kernel.
    




    inf



## 2.2 梯度下降法，求带正则项的代价函数的最小 θ


```python
def grad_descent(X, Y, λ=0, α=0.01, cycleNum=1000):
    """
    输入参数：
        X：mxn 矩阵，np.array类型，数据集里的特征数据
        Y：mx1 列向量
        α：学习率
        n：最大迭代次数
        λ：正则项系数
    """
    m, n = X.shape
    X = np.hstack([np.ones((m, 1)), X])      # 为X增加偏置列
    XT = X.T                                 # 转置X
    weights = np.ones((n+1, 1))              # 模型系数向量 weights，nx1

    for k in range(cycleNum):               # 不建议使用 np.matrix类
        h = sigmoid(np.matmul(X, weights))   # 矩阵乘法，建议使用 np.matmul：  m x 3 ，3 x 1
        weights = weights - α * (np.matmul(XT, h - Y) - λ*weights)
        
    return weights
```

## 2.3 分类器模型


```python
def classifier(X, weights):
    m, n = X.shape
    X = np.hstack([np.ones((m, 1)), X])      # 为X增加偏置列
    
    exp1 = np.exp(np.matmul(X, weights[0]))
    exp2 = np.exp(np.matmul(X, weights[1]))
    
    categories = []
    for e1, e2 in zip(exp1, exp2):
        if not (e1 + e2):
            category = 3
        elif not e1:
            category = 2
        elif not e2:
            category = 1
        categories.append(category)

    return categories
```

## 2.4 评估函数


```python
def evaluate(reallabel, predlabel):
    sum = 0
    for r, p in zip(reallabel, predlabel):
        if r == p:
            sum += 1
    wrongNum = len(reallabel) - sum
    return sum/len(reallabel), wrongNum
```

# 3. 模型训练与评估

## 3.1 使用数据集-1 训练

**划分X, Y：**


```python
x_train = dataset_1.values[:, 1:]
y_train = dataset_1.values[:, 0:1]   # 取出单独一列时，要保证是列向量，不然容易出问题
print(x_train.shape, y_train.shape)
```

    (122, 13) (122, 1)
    

**多次训练，得到各类对应的分类器：**分三类问题，只需要2个分类器


```python
m, n = x_train.shape
weights = []
for i in range(2):   # 分三类问题，只需要2个分类器
    y_bin = np.zeros_like(y_train)
    for j in range(m):
        y_bin[j] = 1 if y_train[j] == i+1 else 0
    # print(f'\n第{i+1}类：\n', y_bin)
    w = grad_descent(x_train, y_bin, λ=1)
    weights.append(w)
print(weights)
```

    d:\program files\python\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: overflow encountered in exp
      
    

    [array([[  -232637.96256243],
           [ -2837410.01882043],
           [  -643457.81721201],
           [  -482567.25135787],
           [ -6211965.0220395 ],
           [-20382262.70092263],
           [  -218194.98862001],
           [   138244.95507542],
           [  -108878.68293053],
           [  -168887.32709772],
           [  -977090.5238757 ],
           [  -164732.81653938],
           [  -257919.52970099],
           [  2717243.55944843]]), array([[  151476.1194002 ],
           [ 1242533.59691288],
           [ -271271.30339635],
           [  186525.64537374],
           [ 2502851.3211024 ],
           [ 8706892.24354161],
           [  340556.37569143],
           [  389264.11050526],
           [   58246.44844866],
           [  277500.59035259],
           [ -845790.4679161 ],
           [  285098.62122438],
           [  580336.98156086],
           [-1476072.38900106]])]
    

## 3.2 使用数据集-2 测试、评估


```python
x_test = dataset_2.values[:, 1:]
y_test = dataset_2.values[:, 0:1]   # 取出单独一列时，要保证是列向量，不然容易出问题
```


```python
y_pred = classifier(x_test, weights)
accuracy, wrongNum = evaluate(y_test, y_pred)
print (f'正确率是：{accuracy}\n识别错误个数：{wrongNum}')
```

    正确率是：0.8
    识别错误个数：11
    

    d:\program files\python\lib\site-packages\ipykernel_launcher.py:5: RuntimeWarning: overflow encountered in exp
      """
    d:\program files\python\lib\site-packages\ipykernel_launcher.py:6: RuntimeWarning: overflow encountered in exp
      
    

## 3.3 对全体数据进行测试


```python
x_test = dataset.values[:, 1:]
y_test = dataset.values[:, 0:1]   # 取出单独一列时，要保证是列向量，不然容易出问题
y_pred = classifier(x_test, weights)
accuracy, wrongNum = evaluate(y_test, y_pred)
print (f'正确率是：{accuracy}\n识别错误个数：{wrongNum}')
```

    正确率是：0.6949152542372882
    识别错误个数：54
    

    d:\program files\python\lib\site-packages\ipykernel_launcher.py:5: RuntimeWarning: overflow encountered in exp
      """
    d:\program files\python\lib\site-packages\ipykernel_launcher.py:6: RuntimeWarning: overflow encountered in exp
      
    
