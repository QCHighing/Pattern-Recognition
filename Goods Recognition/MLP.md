

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
r = clf.score(train_X, train_Y)
print("R值(准确率):", r)  # R值(准确率): 0.8149771726394893
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

## 提交得分：0.60535
## 排名：第17名
