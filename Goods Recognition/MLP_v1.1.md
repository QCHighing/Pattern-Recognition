

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
```


```python
train_set = pd.read_csv('train.csv')
train_X = train_set.values[:, 1:-1]
train_Y = train_set.values[:, -1:]
_, nd = train_X.shape; nd  # 93
```


```python
# 对数据的训练集进行标准化
ss = StandardScaler()
train_X = ss.fit_transform(np.float64(train_X))
```


```python
# 随机生成隐含层size
hl_sizes = []
for i in range(10):
    n = nd // 10 -1
    item = []
    for j in range(1, n):
        p = np.random.rand()
        if p < 0.6:
            a = 5+10*j
            b = a+10*j+1
            if b > nd-5:
                break
            subitem = np.random.randint(a, b)
            if subitem not in item:
                item.append(subitem)
    if item and len(item)>2:
        hl_sizes.append(sorted(item, reverse=True))
hl_sizes.append([80, 60, 40, 20])
hl_sizes
```

    [[50, 46, 42, 22],
     [79, 28, 23],
     [74, 63, 37, 20],
     [78, 52, 43, 19],
     [80, 60, 40, 20]]




```python
# 构建模型
clf = MLPClassifier(solver='lbfgs', random_state=1)
# 自动调参
param_grid = {'alpha':[3, 2, 1, 1e-1, 1e-2],'hidden_layer_sizes':hl_sizes}
grid_search = GridSearchCV(clf, param_grid, n_jobs = 1, verbose=10)
grid_search.fit(train_X[5000:6000,:], train_Y[5000:6000])
alpha, hl_sizes = grid_search.best_params_['alpha'], grid_search.best_params_['hidden_layer_sizes'];alpha,hl_sizes
```


    [CV] alpha=3, hidden_layer_sizes=[50, 46, 42, 22] ....................
    [CV]  alpha=3, hidden_layer_sizes=[50, 46, 42, 22], score=0.6577380952380952, total=   2.6s
    [CV] alpha=3, hidden_layer_sizes=[50, 46, 42, 22] ....................
    [CV]  alpha=3, hidden_layer_sizes=[50, 46, 42, 22], score=0.7275449101796407, total=   2.7s
    [CV] alpha=3, hidden_layer_sizes=[50, 46, 42, 22] ....................
    [CV]  alpha=3, hidden_layer_sizes=[50, 46, 42, 22], score=0.6545454545454545, total=   3.0s

    [CV] alpha=3, hidden_layer_sizes=[79, 28, 23] ........................
    [CV]  alpha=3, hidden_layer_sizes=[79, 28, 23], score=0.6815476190476191, total=   2.5s
    [CV] alpha=3, hidden_layer_sizes=[79, 28, 23] ........................
    [CV]  alpha=3, hidden_layer_sizes=[79, 28, 23], score=0.6976047904191617, total=   2.5s
    [CV] alpha=3, hidden_layer_sizes=[79, 28, 23] ........................
    [CV]  alpha=3, hidden_layer_sizes=[79, 28, 23], score=0.6757575757575758, total=   2.5s

    ... ...

    (0.1, [74, 63, 37, 20])


```python
# 自动调参后，再次构建模型
clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=hl_sizes, random_state=1)
# 训练模型
clf.fit(train_X, train_Y)
r = clf.score(train_X, train_Y)
print("R值(准确率):", r)  # 0.8601268635610683
```

```python
# 预测
test_set = pd.read_csv('test.csv')
test_X = test_set.values[:, 1:]
test_X = ss.fit_transform(np.float64(test_X))
test_Y_predict = clf.predict_proba(test_X)
```


```python
# 写入测试表
submission = pd.read_csv('sampleSubmission.csv')
submission.iloc[:,1:] = test_Y_predict[:]
submission.head()
submission.to_csv('sampleSubmission.csv',sep=',', header=True, index=False)
```

## 提交得分：0.57771
## 排名：第13名
