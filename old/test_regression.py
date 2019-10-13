print('开始导入库...')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as pk
from sklearn.svm import SVR
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.ensemble import GradientBoostingRegressor as GBR
print('导入成功！')

FEATURE_PATH = './nn_model/preconditioned_feature_all.csv'
LABEL_PATH = './nn_model/label_all.csv'
C = 4
GAMMA = 16

# 读入数据
X_train = pd.read_csv(FEATURE_PATH,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)

raw_dataset = pd.read_csv(LABEL_PATH,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)
y_train = raw_dataset['C']

model = SVR(C=C, gamma=GAMMA, kernel='rbf')
model.fit(X_train, y_train)
predicts = model.predict(X_train)



scores = cross_val_score(model, X_train, y_train, cv=3, scoring='explained_variance')
print(scores)

# x = np.array(X_train)
# print(x)
# x = x.reshape(1, -1)

# plt.figure()
# plt.scatter(X_train, y_train, color='g')
# plt.plot(X_train, predicts, color='r')
# plt.show()