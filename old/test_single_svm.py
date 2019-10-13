"""
该模块对给定的数据集进行SVM分类
输入：
C 给定的C
GAMMA 给定的gamma
DATA_PATH 数据集路径
输出：
95%置信区间的正确率
"""

print('开始导入库...')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as pk
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
print('导入成功！')

# SVM超参数
C = 4
GAMMA = 0.158
DATA_PATH = './wine.csv'

# 读入数据
print(DATA_PATH)
raw_dataset = pd.read_csv(DATA_PATH,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)
print(type(raw_dataset), len(raw_dataset))
raw_dataset = raw_dataset.dropna()
label_name = raw_dataset.columns.values.tolist()[-1]
y_train = raw_dataset.pop(label_name)
X_train = raw_dataset.copy()

# 训练和评估
model = SVC(C=C, kernel='rbf', gamma=GAMMA)
scores = cross_val_score(model, X_train, y_train, cv=5)
print(scores)
print("95/100置信区间Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
print()