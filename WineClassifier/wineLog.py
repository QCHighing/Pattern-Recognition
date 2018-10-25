"""
data set used from : https://archive.ics.uci.edu/ml/datasets/wine
"""
import numpy as np
import pandas as pd
from collections import Counter
import math as m

# reading wine.csv and creating a dataframe
filename = "wine.csv"
df = pd.DataFrame(pd.read_table(filename, sep=',', dtype=float))
num_tset = len(df)

# organinzing data
attr = ["class", "alcohol", "malic acid", "ash", "alcalinity of ash", "magnesium",
        "total phenols", "flavanoids", "nonflavanoid phenols", "proanthocyanins",
        "color intensity", "hue", "diluted wines", "proline"]
num_attr = len(attr)

# finding classes
class_votes = Counter(df['class'])
quality = []
for element in class_votes:
    quality.append(element)
del class_votes
num_quality = len(quality)

dataset = []
coeff = np.zeros(num_attr)

# feature scaling all values to 0 to 1
# if all values are already in the range of 0 to 1 we don't scale them further
scale = [1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
for i in range(1, num_attr):
    if max(df[attr[i]]) > 1.0:
        scale[i] = max(df[attr[i]])
    else:
        scale[i] = 1.0

# this is fastest learning rate possible for this data
alpha = [1.0, 1.0, 1.0]


def sigmoidThetaX(x):
    return 1.0 / (1 + m.exp(-1 * coeff.dot(x)))


# input from user
param = []
param.append(1.0)
prompt = "%s : "
par = 0.0
for i in range(1, num_attr):
    par = float(input(prompt % attr[i])) / scale[i]
    param.append(par)

param = np.array(param)


def costFunction():
    error = 0.0
    for i in range(0, num_tset):
        row = []
        # y = 1 or 0
        row.append(dataset[i][0])
        # bias
        row.append(1.0)
        for j in range(1, num_attr):
            row.append(dataset[i][j])
        row = np.array(row)

        x = row[1:]
        thetax = sigmoidThetaX(x)

        error = error + row[0] * m.log(thetax) + (1 - row[0]) * m.log(1 - thetax)

    error = error / (-1.0 * num_tset)
    return error


def calcDerivative(k):
    der = 0.0
    for i in range(0, num_tset):
        row = []
        # y = 1 or 0
        row.append(dataset[i][0])
        # bias
        row.append(1.0)
        for j in range(1, num_attr):
            row.append(dataset[i][j])
        row = np.array(row)

        x = row[1:]
        thetax = sigmoidThetaX(x)

        der = der + (thetax - row[0]) * x[k]

    der = der / num_tset
    return der


def gradientDescent(k):
    tcoeff = []
    val = 0.0
    for i in range(0, num_attr):
        val = coeff[i] - alpha[k] * calcDerivative(i)
        tcoeff.append(val)
    # simultaneous updation
    for i in range(0, num_attr):
        coeff[i] = tcoeff[i]


def logRegression(i):
    J = []
    while True:
        J.append(costFunction())
        res = len(J)
        if J[res - 1] <= 1e-1:
            break
        else:
            gradientDescent(i)

    del J
    y = sigmoidThetaX(param)
    return y


prob = []
count_down = "epoch left: %i"
for i in range(0, num_quality):
    for j in range(0, num_tset):
        temp = []
        val = 0.0
        for k in range(0, num_attr):
            val = df[attr[k]][j] / scale[k]
            temp.append(val)

        if temp[0] == quality[i]:
            temp[0] = 1
        else:
            temp[0] = 0

        dataset.append(temp)
        del temp
    print(count_down % (num_quality - i))
    prob.append([quality[i], logRegression(i)])
    dataset = []

# finding maximum probability
max_prob = prob[0][1]
index = 0
for i in range(1, num_quality):
    if prob[i][1] > max_prob:
        max_prob = prob[i][1]
        index = i

print("quality: ", int(quality[index]))
