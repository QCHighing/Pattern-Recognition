

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
```


```python
train_set = pd.read_csv('train.csv')
train_X = train_set.values[:, 1:-1]
train_Y = train_set.values[:, -1:]
```


```python
# 对数据的训练集进行标准化
ss = StandardScaler()
train_X = ss.fit_transform(np.float64(train_X))
```


```python
# 训练模型
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(90,70,50,30,10), random_state=1)
clf.fit(train_X, train_Y)
```

    d:\program files\python\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:916: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    


```python
r = clf.score(train_X, train_Y)
print("R值(准确率):", r)
```

    R值(准确率): 0.8149771726394893
    


```python
# 预测
test_set = pd.read_csv('test.csv')
test_X = test_set.values[:, 1:]
test_X = ss.fit_transform(np.float64(test_X))
test_Y_predict = clf.predict_proba(test_X)
```


```python
data = pd.DataFrame(test_Y_predict)
data.to_csv('test_results.csv',sep=',', header=True, index=True)
```
