

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
```


```python
train_set = pd.read_csv('train.csv')
train_X = train_set.values[:, 1:-1]   # shape = (49502, 93)
train_Y = train_set.values[:, -1:]

test_set = pd.read_csv('test.csv')
test_X = test_set.values[:, 1:]

train_X.shape, test_X.shape  # (49502, 93), (12376, 93)
```


```python
train_X = train_X.astype(float)
test_X = test_X.astype(float)
X = np.vstack([train_X, test_X])
X.shape  # (61878, 93)
```






```python
# PCA分析
meanValue = np.mean(X, axis=0)              # 计算各特征的均值，按列计算
covMat = np.cov(X, rowvar=0)                # 以列向量为变量计算协方差矩阵
eigVals, eigVects = np.linalg.eig(covMat)   # 计算特征值和特征向量
eigVal_index = np.argsort(eigVals)[::-1]    # 特征值从大到小排序

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

    第1个特征值：62.661609536822915，占比11.772071841190309%
    第2个特征值：45.85441680226809，占比8.614548729632794%
    第3个特征值：34.00380893908432，占比6.388206186593045%
    ... ...
    第55个特征值：1.4717337577048528，占比0.2764907517516815%
    第56个特征值：1.456902230879997，占比0.27370439180039197%
    第57个特征值：1.4021227030840104，占比0.26341310593321554%

    前57个特征值满足占比 95% 以上！共有93个特征值



```python
# PCA降维
eigVectsMain = eigVects[:, eigVal_index[:k]]   # 选取基向量
X_ld = np.matmul(X, eigVectsMain)  # 降维
train_X_ld = X_ld[:49502]
test_X_ld = X_ld[49502:]
_, nd = test_X_ld.shape; n  # 57
```




```python
# 对数据的训练集进行标准化
ss = StandardScaler()
train_X = ss.fit_transform(np.float64(train_X_ld))
```


```python
# 随机生成隐含层尺寸
hl_sizes = []
for i in range(7):
    n = nd // 10 -1
    item = []
    for j in range(1,n):
        p = np.random.rand()
        if p < 0.6:
            a = 9+10*j
            b = a+10*j+1
            if b > nd-5:
                break
            item.append(np.random.randint(a, b))
    if item:
        hl_sizes.append(sorted(item, reverse=True))
hl_sizes.append([40,20])
hl_sizes  # [[43], [43, 28], [46, 28], [24], [36, 23], [42, 24], [39], [40, 20]]
```


```python
# 构建模型
clf = MLPClassifier(solver='lbfgs', random_state=1)
# 自动调参
param_grid = {'alpha':[1, 1e-1, 1e-2, 1e-3, 1e-4],'hidden_layer_sizes':hl_sizes}
grid_search = GridSearchCV(clf, param_grid, n_jobs = 1, verbose=10)
grid_search.fit(train_X[8000:12000,:], train_Y[8000:12000])
alpha, hl_sizes = grid_search.best_params_['alpha'], grid_search.best_params_['hidden_layer_sizes'];alpha,hl_sizes
```

    [CV] alpha=1, hidden_layer_sizes=[43] ................................
    [CV]  alpha=1, hidden_layer_sizes=[43], score=0.7095808383233533, total=   2.7s
    [CV] alpha=1, hidden_layer_sizes=[43] ................................
    [CV]  alpha=1, hidden_layer_sizes=[43], score=0.6849212303075769, total=   3.0s
    [CV] alpha=1, hidden_layer_sizes=[43] ................................
    [CV]  alpha=1, hidden_layer_sizes=[43], score=0.7175056348610067, total=   3.1s

    [CV] alpha=1, hidden_layer_sizes=[43, 28] ............................
    [CV]  alpha=1, hidden_layer_sizes=[43, 28], score=0.6841317365269461, total=   4.0s
    [CV] alpha=1, hidden_layer_sizes=[43, 28] ............................
    [CV]  alpha=1, hidden_layer_sizes=[43, 28], score=0.6894223555888972, total=   3.9s
    [CV] alpha=1, hidden_layer_sizes=[43, 28] ............................
    [CV]  alpha=1, hidden_layer_sizes=[43, 28], score=0.6994740796393689, total=   4.0s

    [CV] alpha=1, hidden_layer_sizes=[46, 28] ............................
    [CV]  alpha=1, hidden_layer_sizes=[46, 28], score=0.686377245508982, total=   4.0s
    [CV] alpha=1, hidden_layer_sizes=[46, 28] ............................
    [CV]  alpha=1, hidden_layer_sizes=[46, 28], score=0.6789197299324832, total=   4.5s
    [CV] alpha=1, hidden_layer_sizes=[46, 28] ............................
    [CV]  alpha=1, hidden_layer_sizes=[46, 28], score=0.6972201352366642, total=   4.6s

    [CV] alpha=1, hidden_layer_sizes=[24] ................................
    [CV]  alpha=1, hidden_layer_sizes=[24], score=0.6908682634730539, total=   1.9s
    [CV] alpha=1, hidden_layer_sizes=[24] ................................
    [CV]  alpha=1, hidden_layer_sizes=[24], score=0.7014253563390848, total=   2.1s
    [CV] alpha=1, hidden_layer_sizes=[24] ................................
    [CV]  alpha=1, hidden_layer_sizes=[24], score=0.6776859504132231, total=   1.8s

    [CV] alpha=1, hidden_layer_sizes=[36, 23] ............................
    [CV]  alpha=1, hidden_layer_sizes=[36, 23], score=0.6841317365269461, total=   3.5s
    [CV] alpha=1, hidden_layer_sizes=[36, 23] ............................
    [CV]  alpha=1, hidden_layer_sizes=[36, 23], score=0.6864216054013503, total=   3.6s
    [CV] alpha=1, hidden_layer_sizes=[36, 23] ............................
    [CV]  alpha=1, hidden_layer_sizes=[36, 23], score=0.6724267468069121, total=   3.5s

    [CV] alpha=1, hidden_layer_sizes=[42, 24] ............................
    [CV]  alpha=1, hidden_layer_sizes=[42, 24], score=0.6781437125748503, total=   3.5s
    [CV] alpha=1, hidden_layer_sizes=[42, 24] ............................
    [CV]  alpha=1, hidden_layer_sizes=[42, 24], score=0.6856714178544636, total=   3.6s
    [CV] alpha=1, hidden_layer_sizes=[42, 24] ............................
    [CV]  alpha=1, hidden_layer_sizes=[42, 24], score=0.6844477836213373, total=   3.5s

    [CV] alpha=1, hidden_layer_sizes=[39] ................................
    [CV]  alpha=1, hidden_layer_sizes=[39], score=0.6931137724550899, total=   2.5s
    [CV] alpha=1, hidden_layer_sizes=[39] ................................
    [CV]  alpha=1, hidden_layer_sizes=[39], score=0.6939234808702176, total=   3.5s
    [CV] alpha=1, hidden_layer_sizes=[39] ................................
    [CV]  alpha=1, hidden_layer_sizes=[39], score=0.7002253944402704, total=   2.8s

    [CV] alpha=1, hidden_layer_sizes=[40, 20] ............................
    [CV]  alpha=1, hidden_layer_sizes=[40, 20], score=0.6691616766467066, total=   8.2s
    [CV] alpha=1, hidden_layer_sizes=[40, 20] ............................
    [CV]  alpha=1, hidden_layer_sizes=[40, 20], score=0.6766691672918229, total=  24.0s
    [CV] alpha=1, hidden_layer_sizes=[40, 20] ............................
    [CV]  alpha=1, hidden_layer_sizes=[40, 20], score=0.6851990984222389, total=  10.2s

    (1, [43])




```python
# 自动调参后，再次构建模型
clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=hl_sizes, random_state=1)
# 训练模型
clf.fit(train_X_ld, train_Y)
r = clf.score(train_X_ld, train_Y)
print("R值(准确率):", r)
```
    R值(准确率): 0.8036644984041049



```python
# 预测
test_X_ld = ss.fit_transform(np.float64(test_X_ld))  # 标准化
test_Y_predict = clf.predict_proba(test_X_ld)
```


```python
# 写入测试表
submission = pd.read_csv('sampleSubmission.csv')
submission.iloc[:,1:] = test_Y_predict[:]
submission.head()
submission.to_csv('sampleSubmission.csv',sep=',', header=True, index=False)
```

## 提交得分：1.23000
