

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
```

# 1. 数据处理

### 1.1 读取单行txt数据


```python
# line = open('dataset.txt').readlines(1)
line = open('secom.txt').readlines(1)
list(line)
```




    ['3030.93 2564 2187.7333 1411.1265 1.3602 100 97.6133 0.1242 1.5005 0.0162 -0.0034 0.9455 202.4396 0 7.9558 414.871 10.0433 0.968 192.3963 12.519 1.4026 -5419 2916.5 -4043.75 751 0.8955 1.773 3.049 64.2333 2.0222 0.1632 3.5191 83.3971 9.5126 50.617 64.2588 49.383 66.3141 86.9555 117.5132 61.29 4.515 70 352.7173 10.1841 130.3691 723.3092 1.3072 141.2282 1 624.3145 218.3174 0 4.592 4.841 2834 0.9317 0.9484 4.7057 -1.7264 350.9264 10.6231 108.6427 16.1445 21.7264 29.5367 693.7724 0.9226 148.6009 1 608.17 84.0793 NaN NaN 0 0.0126 -0.0206 0.0141 -0.0307 -0.0083 -0.0026 -0.0567 -0.0044 7.2163 0.132 NaN 2.3895 0.969 1747.6049 0.1841 8671.9301 -0.3274 -0.0055 -0.0001 0.0001 0.0003 -0.2786 0 0.3974 -0.0251 0.0002 0.0002 0.135 -0.0042 0.0003 0.0056 0 -0.2468 0.3196 NaN NaN NaN NaN 0.946 0 748.6115 0.9908 58.4306 0.6002 0.9804 6.3788 15.88 2.639 15.94 15.93 0.8656 3.353 0.4098 3.188 -0.0473 0.7243 0.996 2.2967 1000.7263 39.2373 123 111.3 75.2 46.2 350.671 0.3948 0 6.78 0.0034 0.0898 0.085 0.0358 0.0328 12.2566 0 4.271 10.284 0.4734 0.0167 11.8901 0.41 0.0506 NaN NaN 1017 967 1066 368 0.09 0.048 0.095 2 0.9 0.069 0.046 0.725 0.1139 0.3183 0.5888 0.3184 0.9499 0.3979 0.16 0 0 20.95 0.333 12.49 16.713 0.0803 5.72 0 11.19 65.363 0 0 0 0 0 0 0.292 5.38 20.1 0.296 10.62 10.3 5.38 4.04 16.23 0.2951 8.64 0 10.3 97.314 0 0.0772 0.0599 0.07 0.0547 0.0704 0.052 0.0301 0.1135 3.4789 0.001 NaN 0.0707 0.0211 175.2173 0.0315 1940.3994 0 0.0744 0.0546 0 0 0 0 0 0 0 0 0 0.0027 0.004 0 0 0 0 NaN NaN NaN NaN 0.0188 0 219.9453 0.0011 2.8374 0.0189 0.005 0.4269 0 0 0 0 0 0 0 0 0 0 0 0.0472 40.855 4.5152 30.9815 33.9606 22.9057 15.9525 110.2144 0.131 0 2.5883 0.001 0.0319 0.0197 0.012 0.0109 3.9321 0 1.5123 3.5811 0.1337 0.0055 3.8447 0.1077 0.0167 NaN NaN 418.1363 398.3185 496.1582 158.333 0.0373 0.0202 0.0462 0.6083 0.3032 0.02 0.0174 0.2827 0.0434 0.1342 0.2419 0.1343 0.367 0.1431 0.061 0 0 0 6.2698 0.1181 3.8208 5.3737 0.0254 1.6252 0 3.2461 18.0118 0 0 0 0 0 0 0.0752 1.5989 6.5893 0.0913 3.0911 8.4654 1.5989 1.2293 5.3406 0.0867 2.8551 0 2.9971 31.8843 NaN NaN 0 0.0215 0.0274 0.0315 0.0238 0.0206 0.0238 0.0144 0.0491 1.2708 0.0004 NaN 0.0229 0.0065 55.2039 0.0105 560.2658 0 0.017 0.0148 0.0124 0.0114 0 0 0 0 0 0 0 0.001 0.0013 0 0 0 0 NaN NaN NaN NaN 0.0055 0 61.5932 0.0003 0.9967 0.0082 0.0017 0.1437 0 0 0 0 0 0 0 0 0 0 0 0.0151 14.2396 1.4392 5.6188 3.6721 2.9329 2.1118 24.8504 29.0271 0 6.9458 2.738 5.9846 525.0965 0 3.4641 6.0544 0 53.684 2.4788 4.7141 1.7275 6.18 3.275 3.6084 18.7673 33.1562 26.3617 49.0013 10.0503 2.7073 3.1158 3.1136 44.5055 42.2737 1.3071 0.8693 1.1975 0.6288 0.9163 0.6448 1.4324 0.4576 0.1362 0 0 0 5.9396 3.2698 9.5805 2.3106 6.1463 4.0502 0 1.7924 29.9394 0 0 0 0 0 0 6.2052 311.6377 5.7277 2.7864 9.7752 63.7987 24.7625 13.6778 2.3394 31.9893 5.8142 0 1.6936 115.7408 0 613.3069 291.4842 494.6996 178.1759 843.1138 0 53.1098 0 48.2091 0.7578 NaN 2.957 2.1739 10.0261 17.1202 22.3756 0 0 0 0 0 0 0 0 0 0 0 0 64.6707 0 0 0 0 0 NaN NaN NaN NaN 1.9864 0 29.3804 0.1094 4.856 3.1406 0.5064 6.6926 0 0 0 0 0 0 0 0 0 0 0 2.057 4.0825 11.5074 0.1096 0.0078 0.0026 7.116 1.0616 395.57 75.752 0.4234 12.93 0.78 0.1827 5.7349 0.3363 39.8842 3.2687 1.0297 1.0344 0.4385 0.1039 42.3877 NaN NaN NaN NaN NaN NaN NaN NaN 533.85 2.1113 8.95 0.3157 3.0624 0.1026 1.6765 14.9509 NaN NaN NaN NaN 0.5005 0.0118 0.0035 2.363 NaN NaN NaN NaN\n']



### 1.2 以表格形式打开txt数据


```python
# dataset = pd.read_csv("dataset.txt",sep="\t", names=["x1","x2"])
colnames = ['x'+str(i) for i in range(590)]
dataset = pd.read_csv("secom.txt",sep=" ", names=colnames)
dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>x9</th>
      <th>...</th>
      <th>x580</th>
      <th>x581</th>
      <th>x582</th>
      <th>x583</th>
      <th>x584</th>
      <th>x585</th>
      <th>x586</th>
      <th>x587</th>
      <th>x588</th>
      <th>x589</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3030.93</td>
      <td>2564.00</td>
      <td>2187.7333</td>
      <td>1411.1265</td>
      <td>1.3602</td>
      <td>100.0</td>
      <td>97.6133</td>
      <td>0.1242</td>
      <td>1.5005</td>
      <td>0.0162</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.5005</td>
      <td>0.0118</td>
      <td>0.0035</td>
      <td>2.3630</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3095.78</td>
      <td>2465.14</td>
      <td>2230.4222</td>
      <td>1463.6606</td>
      <td>0.8294</td>
      <td>100.0</td>
      <td>102.3433</td>
      <td>0.1247</td>
      <td>1.4966</td>
      <td>-0.0005</td>
      <td>...</td>
      <td>0.0060</td>
      <td>208.2045</td>
      <td>0.5019</td>
      <td>0.0223</td>
      <td>0.0055</td>
      <td>4.4447</td>
      <td>0.0096</td>
      <td>0.0201</td>
      <td>0.0060</td>
      <td>208.2045</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2932.61</td>
      <td>2559.94</td>
      <td>2186.4111</td>
      <td>1698.0172</td>
      <td>1.5102</td>
      <td>100.0</td>
      <td>95.4878</td>
      <td>0.1241</td>
      <td>1.4436</td>
      <td>0.0041</td>
      <td>...</td>
      <td>0.0148</td>
      <td>82.8602</td>
      <td>0.4958</td>
      <td>0.0157</td>
      <td>0.0039</td>
      <td>3.1745</td>
      <td>0.0584</td>
      <td>0.0484</td>
      <td>0.0148</td>
      <td>82.8602</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2988.72</td>
      <td>2479.90</td>
      <td>2199.0333</td>
      <td>909.7926</td>
      <td>1.3204</td>
      <td>100.0</td>
      <td>104.2367</td>
      <td>0.1217</td>
      <td>1.4882</td>
      <td>-0.0124</td>
      <td>...</td>
      <td>0.0044</td>
      <td>73.8432</td>
      <td>0.4990</td>
      <td>0.0103</td>
      <td>0.0025</td>
      <td>2.0544</td>
      <td>0.0202</td>
      <td>0.0149</td>
      <td>0.0044</td>
      <td>73.8432</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3032.24</td>
      <td>2502.87</td>
      <td>2233.3667</td>
      <td>1326.5200</td>
      <td>1.5334</td>
      <td>100.0</td>
      <td>100.3967</td>
      <td>0.1235</td>
      <td>1.5031</td>
      <td>-0.0031</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.4800</td>
      <td>0.4766</td>
      <td>0.1045</td>
      <td>99.3032</td>
      <td>0.0202</td>
      <td>0.0149</td>
      <td>0.0044</td>
      <td>73.8432</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 590 columns</p>
</div>



### 1.3 存储为csv文件


```python
# dataset.to_csv('dataset.csv', index=False, sep=',')
dataset.to_csv('secom.csv', index=False, sep=',')
```

### 1.4 打开csv文件


```python
# dataset = pd.read_csv("dataset.csv")
dataset = pd.read_csv("secom.csv")
dataset.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>x9</th>
      <th>...</th>
      <th>x580</th>
      <th>x581</th>
      <th>x582</th>
      <th>x583</th>
      <th>x584</th>
      <th>x585</th>
      <th>x586</th>
      <th>x587</th>
      <th>x588</th>
      <th>x589</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1562</th>
      <td>2899.41</td>
      <td>2464.36</td>
      <td>2179.7333</td>
      <td>3085.3781</td>
      <td>1.4843</td>
      <td>100.0</td>
      <td>82.2467</td>
      <td>0.1248</td>
      <td>1.3424</td>
      <td>-0.0045</td>
      <td>...</td>
      <td>0.0047</td>
      <td>203.1720</td>
      <td>0.4988</td>
      <td>0.0143</td>
      <td>0.0039</td>
      <td>2.8669</td>
      <td>0.0068</td>
      <td>0.0138</td>
      <td>0.0047</td>
      <td>203.1720</td>
    </tr>
    <tr>
      <th>1563</th>
      <td>3052.31</td>
      <td>2522.55</td>
      <td>2198.5667</td>
      <td>1124.6595</td>
      <td>0.8763</td>
      <td>100.0</td>
      <td>98.4689</td>
      <td>0.1205</td>
      <td>1.4333</td>
      <td>-0.0061</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.4975</td>
      <td>0.0131</td>
      <td>0.0036</td>
      <td>2.6238</td>
      <td>0.0068</td>
      <td>0.0138</td>
      <td>0.0047</td>
      <td>203.1720</td>
    </tr>
    <tr>
      <th>1564</th>
      <td>2978.81</td>
      <td>2379.78</td>
      <td>2206.3000</td>
      <td>1110.4967</td>
      <td>0.8236</td>
      <td>100.0</td>
      <td>99.4122</td>
      <td>0.1208</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0025</td>
      <td>43.5231</td>
      <td>0.4987</td>
      <td>0.0153</td>
      <td>0.0041</td>
      <td>3.0590</td>
      <td>0.0197</td>
      <td>0.0086</td>
      <td>0.0025</td>
      <td>43.5231</td>
    </tr>
    <tr>
      <th>1565</th>
      <td>2894.92</td>
      <td>2532.01</td>
      <td>2177.0333</td>
      <td>1183.7287</td>
      <td>1.5726</td>
      <td>100.0</td>
      <td>98.7978</td>
      <td>0.1213</td>
      <td>1.4622</td>
      <td>-0.0072</td>
      <td>...</td>
      <td>0.0075</td>
      <td>93.4941</td>
      <td>0.5004</td>
      <td>0.0178</td>
      <td>0.0038</td>
      <td>3.5662</td>
      <td>0.0262</td>
      <td>0.0245</td>
      <td>0.0075</td>
      <td>93.4941</td>
    </tr>
    <tr>
      <th>1566</th>
      <td>2944.92</td>
      <td>2450.76</td>
      <td>2195.4444</td>
      <td>2914.1792</td>
      <td>1.5978</td>
      <td>100.0</td>
      <td>85.1011</td>
      <td>0.1235</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0045</td>
      <td>137.7844</td>
      <td>0.4987</td>
      <td>0.0181</td>
      <td>0.0040</td>
      <td>3.6275</td>
      <td>0.0117</td>
      <td>0.0162</td>
      <td>0.0045</td>
      <td>137.7844</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 590 columns</p>
</div>



### 1.5 填充NaN值


```python
pd.isnull(dataset).values.any()        # 检查是否存在NaN值
```




    True

#### 1.5.1 以列-众数填充，速度最慢


```python
imp = SimpleImputer(strategy='most_frequent')
imp.fit(dataset)
dataset_nn = imp.transform(dataset)  # 返回 numpy.ndarray 类型dataset_nn
dataset_nn[:5]
```




    array([[3.0309300e+03, 2.5640000e+03, 2.1877333e+03, ..., 1.1200000e-02,
            3.3000000e-03, 0.0000000e+00],
           [3.0957800e+03, 2.4651400e+03, 2.2304222e+03, ..., 2.0100000e-02,
            6.0000000e-03, 2.0820450e+02],
           [2.9326100e+03, 2.5599400e+03, 2.1864111e+03, ..., 4.8400000e-02,
            1.4800000e-02, 8.2860200e+01],
           [2.9887200e+03, 2.4799000e+03, 2.1990333e+03, ..., 1.4900000e-02,
            4.4000000e-03, 7.3843200e+01],
           [3.0322400e+03, 2.5028700e+03, 2.2333667e+03, ..., 1.4900000e-02,
            4.4000000e-03, 7.3843200e+01]])


#### 1.5.2 以列-中位数填充

```python
imp = SimpleImputer(strategy='median')
imp.fit(dataset)
dataset_nn = imp.transform(dataset)  # 返回 numpy.ndarray 类型dataset_nn
dataset_nn[:5]
```




    array([[3.0309300e+03, 2.5640000e+03, 2.1877333e+03, ..., 1.4800000e-02,
            4.6000000e-03, 7.1900500e+01],
           [3.0957800e+03, 2.4651400e+03, 2.2304222e+03, ..., 2.0100000e-02,
            6.0000000e-03, 2.0820450e+02],
           [2.9326100e+03, 2.5599400e+03, 2.1864111e+03, ..., 4.8400000e-02,
            1.4800000e-02, 8.2860200e+01],
           [2.9887200e+03, 2.4799000e+03, 2.1990333e+03, ..., 1.4900000e-02,
            4.4000000e-03, 7.3843200e+01],
           [3.0322400e+03, 2.5028700e+03, 2.2333667e+03, ..., 1.4900000e-02,
            4.4000000e-03, 7.3843200e+01]])


#### 1.5.3 默认以列-平均数填充

```python
imp = SimpleImputer()
imp.fit(dataset)
dataset_nn = imp.transform(dataset)  # 返回 numpy.ndarray 类型dataset_nn
dataset_nn[:5]
```




    array([[3.03093000e+03, 2.56400000e+03, 2.18773330e+03, ...,
            1.64749042e-02, 5.28333333e-03, 9.96700663e+01],
           [3.09578000e+03, 2.46514000e+03, 2.23042220e+03, ...,
            2.01000000e-02, 6.00000000e-03, 2.08204500e+02],
           [2.93261000e+03, 2.55994000e+03, 2.18641110e+03, ...,
            4.84000000e-02, 1.48000000e-02, 8.28602000e+01],
           [2.98872000e+03, 2.47990000e+03, 2.19903330e+03, ...,
            1.49000000e-02, 4.40000000e-03, 7.38432000e+01],
           [3.03224000e+03, 2.50287000e+03, 2.23336670e+03, ...,
            1.49000000e-02, 4.40000000e-03, 7.38432000e+01]])



### 1.6 归一化（可选）


```python
dataset_nn_scale = preprocessing.scale(dataset_nn)
```

### 1.7 数据示图


```python
plt.plot(dataset_nn[:, 0], dataset_nn[:, 1], 'rs')
plt.title("2D data")
```




    Text(0.5, 1.0, '2D data')


![iTekrt.png](https://s1.ax1x.com/2018/11/06/iTekrt.png)



# 2. 主成分分析

### 2.1 计算特征均值


```python
meanValue = np.mean(dataset_nn, axis=0)  # 按列计算均值
```

### 2.2 计算数据集的协方差矩阵


```python
covMat = np.cov(dataset_nn, rowvar=0)    # 以列向量为变量计算协方差矩阵
```

### 2.3 计算协方差矩阵的特征值与特征向量


```python
eigVals, eigVects = np.linalg.eig(covMat)
# print(f'eigVals=\n{eigVals} \n\n eigVects=\n{eigVects}')
```

### 2.4 特征值/特征向量排序


```python
eigVal_index = np.argsort(eigVals)[::-1]  # 从大到小排序
```

### 2.5 根据特征值占比选取主成分


```python
sum_eigVals = sum(eigVals)
threshold = sum_eigVals * 0.95
tempsum = 0
for i in range(len(eigVals)):
    v = eigVals[eigVal_index[i]]
    tempsum += v
    print(f'第{i+1}个特征值：{v}，占比{v/sum_eigVals*100}%')
    if tempsum >= threshold:
        k = i + 1
        print(f'\n前{i+1}个特征值满足占比 95% 以上！共有{len(eigVals)}个特征值')
        break
```

    第1个特征值：53415197.856875174，占比59.25405798334119%
    第2个特征值：21746671.90465919，占比24.12381886960505%
    第3个特征值：8248376.615290736，占比9.150013588650685%
    第4个特征值：2073880.8592939659，占比2.3005785173054307%
    第5个特征值：1315404.3877582948，占比1.459192345830452%

    前5个特征值满足占比 95% 以上！共有590个特征值



```python
eigVal_index[:k]
```




    array([0, 1, 2, 3, 4], dtype=int64)



### 2.6 选取基向量


```python
eigVectsMain = eigVects[:, eigVal_index[:k]] ; eigVectsMain.shape
```




    (590, 5)



### 2.7 降维


```python
lowDDataMat = np.matmul(dataset_nn, eigVectsMain)
```

### 2.8 还原特征


```python
reconMat = np.matmul(lowDDataMat, eigVectsMain.T) + meanValue
```


```python
plt.plot(reconMat[:, 0], reconMat[:, 1], 'rs')
plt.title("2D data")
```




    Text(0.5, 1.0, '2D data')


![iTeAqP.png](https://s1.ax1x.com/2018/11/06/iTeAqP.png)
