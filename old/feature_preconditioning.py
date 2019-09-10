"""
此模块进行数据预处理
标准化 + PCA
输入：
DATASET_PATH 数据集路径
输出：
./StantardData/ 标准化后的数据集放入，
./PreconditionedData/ 降维后的数据集放入
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 使用PCA将属性降维至
REDUCE_TO = 4

# 数据集路径
DATASET_PATH = './nn_model/feature.csv'
SAVE_TO = './nn_model/preconditioned_feature.csv'


# 读入数据
X_train = pd.read_csv(DATASET_PATH,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)

# 标准化
stat = X_train.describe().transpose()
mean = stat['mean']
std = stat['std']
for i in X_train:
	if std[i] != 0:
		X_train[i] = (X_train[i] - mean[i]) / std[i]
	else:
		X_train[i] = X_train[i] - mean[i]
X_train.to_csv(SAVE_TO, index=False)

# # PCA
# pca = PCA(REDUCE_TO)
# X_train = pd.DataFrame(pca.fit_transform(X_train))
# X_train.to_csv(SAVE_TO, index=False)

print('预处理成功！')
