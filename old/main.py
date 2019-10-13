'''
输入：
多个待参数寻优的数据集路径
神经网络模型位置
网格算法参数
PCA降至的维数
REGRETION_METHOD 1为神经网络拟合，2为SVR
完成：
数据预处理
特征计算
（训练神经网络以外的回归模型）
带入模型，得到较优参数
网格搜索得到最优参数
输出：
较优参数SVM评估
最优参数SVM评估
'''

print('开始导入库...')
import cbz_method
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import time
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
print('导入成功！')

GRID_STRIDE = 1
UPPER_BOUND = 10
LOWER_BOUND = -10
REDUCE_TO = 3
# FEATURE_REDUCE_TO = 4
C_MODEL_PATH = './nn_model/model_c.h5'
GAMMA_MODEL_PATH = './nn_model/model_gamma.h5'
REGRETION_METHOD = 2
EPOCH = 100
DATASET_PATH = [
	'./PreconditionedData/BinaryClassificationDatabase1/breast.csv',
	'./PreconditionedData/BinaryClassificationDatabase1/breast-cancer(diagnostic).csv',
	'./PreconditionedData/BinaryClassificationDatabase1/breast-cancer.csv',
	'./PreconditionedData/BinaryClassificationDatabase1/climate-model.csv',
	'./PreconditionedData/BinaryClassificationDatabase1/connectionist.csv',
	'./PreconditionedData/BinaryClassificationDatabase1/extention.csv',
	'./PreconditionedData/BinaryClassificationDatabase1/fertility.csv',
	'./PreconditionedData/BinaryClassificationDatabase1/ilpd.csv',
	'./PreconditionedData/BinaryClassificationDatabase1/monk (1).csv',
	'./PreconditionedData/BinaryClassificationDatabase1/monk (2).csv',
	'./PreconditionedData/BinaryClassificationDatabase1/monk (3).csv',
	'./PreconditionedData/BinaryClassificationDatabase1/spectf.csv',
	'./PreconditionedData/BinaryClassificationDatabase1/thoracic.csv',
	'./PreconditionedData/BinaryClassificationDatabase1/wine.csv',
]

# 网格搜索数据准备
big_array = []
i = LOWER_BOUND
while i <= UPPER_BOUND + 0.001:
	big_array.append(2**i)
	i += GRID_STRIDE

time_of_grid = []
time_of_us = []

for dataset in DATASET_PATH:
	print(dataset, ": ")

	''' 数据预处理 '''
	# 读入数据


	raw_dataset = pd.read_csv(dataset,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)
	raw_dataset = raw_dataset.dropna()
	label_name = raw_dataset.columns.values.tolist()[-1]
	y_train = raw_dataset.pop(label_name)
	X_train = raw_dataset.copy()


	# 标准化
	stat = X_train.describe().transpose()
	mean = stat['mean']
	std = stat['std']
	for i in X_train:
		if std[i] != 0:
			X_train[i] = (X_train[i] - mean[i]) / std[i]
		else:
			X_train[i] = X_train[i] - mean[i]
	X_train['Label'] = y_train

	# PCA
	X_train.pop('Label')
	pca = PCA(REDUCE_TO)
	X_train = pd.DataFrame(pca.fit_transform(X_train))
	# X_train['Label'] = y_train

	''' 特征计算 '''
	feature = [stat['std'][0], stat['std'][1], stat['std'][2],
	stat.skew()[0], stat.skew()[1], stat.skew()[2]]
	feature = np.expand_dims(feature, 0)
	print("feature:")
	print(feature)

	t2 = time.time()

	''' 放入模型，计算较优参数 '''
	model_c = keras.models.load_model('./nn_model/model_c.h5')
	# model_c.summary()
	predict_c = model_c.predict(feature)

	model_gamma = keras.models.load_model('./nn_model/model_gamma.h5')
	# model_gamma.summary()
	predict_gamma = model_gamma.predict(feature)

	log2_C = predict_c[0][0] 			# 较优log2_C
	log2_gamma = predict_gamma[0][0] 	# 较优log2_gamma
	predicted_C = 2**log2_C
	predicted_gamma = 2**log2_gamma

	print("Predicted C is", predicted_C)
	print("Predicted gamma is", predicted_gamma)
	model = SVC(C=predicted_C, kernel='rbf', gamma=predicted_gamma)
	scores = cross_val_score(model, X_train, y_train, cv=5)
	print("95/100置信区间Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

	''' 找局部最优 '''
	data = X_train.copy()
	data['Label'] = y_train
	line_num = len(data)
	# print(line_num)
	sample_num = int(0.8*line_num)
	# print(sample_num)
	for i in range(0, EPOCH+1):
		data = data.sample(frac=1) # 打乱
		X_train = data.iloc[0:sample_num]
		X_test = data.iloc[sample_num:line_num+1]
		y_train = X_train.pop('Label')
		y_test = X_test.pop('Label')
		# print(X_train.head())
		# print(X_test.head())

		log2_C, log2_gamma = cbz_method.optimize(X_train, y_train, X_test, y_test, log2_C, log2_gamma)

	predicted_C = 2**log2_C
	predicted_gamma = 2**log2_gamma
	print("Finally, Predicted C is", predicted_C)
	print("Finally, Predicted gamma is", predicted_gamma)
	model = SVC(C=predicted_C, kernel='rbf', gamma=predicted_gamma)
	scores = cross_val_score(model, X_train, y_train, cv=5)
	print("95/100置信区间Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
	print()

	t3 = time.time()

	''' 网格搜索获得最优参数 '''
	param_grid = [{
		'C': big_array,
		'gamma': big_array,
		'kernel': ['rbf']
	}]
	model = SVC()
	gs_model = GridSearchCV(model, param_grid, cv=5, iid=True)
	gs_model.fit(X_train, y_train)

	best_C = gs_model.best_params_['C']			# 最优C
	best_gamma = gs_model.best_params_['gamma']	# 最优gamma

	''' 输出评估结果 '''
	print("Best C is", best_C)
	print("Best gamma is", best_gamma)
	print("Best score is", gs_model.best_score_)

	t4 = time.time()

	time_of_us.append(t3-t2)
	time_of_grid.append(t4-t3)
	print(time_of_us)
	print(time_of_grid)


