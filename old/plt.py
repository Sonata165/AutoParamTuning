"""
该模块用来画图，将数据集特征和最优参数的关系可视化
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FEATURE_PATH = './nn_model/feature.csv'
LABEL_PATH = './nn_model/label.csv'

# 读入数据
X_train = pd.read_csv(FEATURE_PATH,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)
print(type(X_train), len(X_train))
y_train = pd.read_csv(LABEL_PATH,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)
print(type(y_train), len(y_train))
print(y_train)

C = y_train['C']
gamma = y_train['gamma']
X_train['C'] = C
X_train['gamma'] = gamma

sb.pairplot(X_train, diag_kind='kde')
# plt.plot(feature1, gamma, 'ro')
plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(feature1, feature2, C)
# plt.show()


# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# ax.set_xlabel('log2(Gamma)')
# ax.set_ylabel('log2(C)')
# plt.show()
