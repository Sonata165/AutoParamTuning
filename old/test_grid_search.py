"""
此模块对某个特定的数据集进行网格搜索，找到最优参数
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
import numpy as np
import pandas as pd
import _pickle as pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
print('导入成功！')

C = 4
GAMMA = 2
GRID_STRIDE = 1
UPPER_BOUND = 10
LOWER_BOUND = -10
DATASET_PATH = [
	'./BinaryClassificationDatabase1/breast.csv',
	'./BinaryClassificationDatabase1/breast-cancer(diagnostic).csv',
	'./BinaryClassificationDatabase1/climate-model.csv',
	'./BinaryClassificationDatabase1/connectionist.csv',
	'./BinaryClassificationDatabase1/extention.csv',
	'./BinaryClassificationDatabase1/fertility.csv',
	'./BinaryClassificationDatabase1/ilpd.csv',
	'./BinaryClassificationDatabase1/monk (1).csv',
	'./BinaryClassificationDatabase1/spectf.csv',
	'./BinaryClassificationDatabase1/thoracic.csv',
	'./BinaryClassificationDatabase1/wine.csv',
]

for DATA_PATH in DATASET_PATH:
	# 读入数据
	raw_dataset = pd.read_csv(DATA_PATH,
		na_values='?', comment='\t', sep=',', skipinitialspace=True)
	print(type(raw_dataset))
	raw_dataset = raw_dataset.dropna()
	print(raw_dataset)
	label_name = raw_dataset.columns.values.tolist()[-1]
	y_train = raw_dataset.pop(label_name)
	X_train = raw_dataset.copy()

	# 网格搜索数据准备
	big_array = []
	i = LOWER_BOUND
	while i <= UPPER_BOUND + 0.001:
		big_array.append(2**i)
		i += GRID_STRIDE


	# 训练
	print(big_array)
	param_grid = [{
		'C': big_array,
		'gamma': big_array,
		'kernel': ['rbf']
	}]
	model = SVC()
	gs_model = GridSearchCV(model, param_grid, cv=5, iid=True)
	gs_model.fit(X_train, y_train)


	# 画图
	# print(gs_model.cv_results_)
	c_arr = gs_model.cv_results_['param_C']
	gamma_arr = gs_model.cv_results_['param_gamma']
	test_score = gs_model.cv_results_['mean_test_score']

	X = np.arange(LOWER_BOUND, UPPER_BOUND+GRID_STRIDE, GRID_STRIDE)
	Y = np.arange(LOWER_BOUND, UPPER_BOUND+GRID_STRIDE, GRID_STRIDE)
	row = len(X)
	X, Y = np.meshgrid(X, Y)
	Z = np.array(test_score).reshape(row, -1)

	fig = plt.figure()
	ax = Axes3D(fig)
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
	ax.set_xlabel('log2(Gamma)')
	ax.set_ylabel('log2(C)')
	plt.show()

	best_C = gs_model.best_params_['C']
	best_gamma = gs_model.best_params_['gamma']
	print('best log2(C) is', + np.log2(best_C))
	print('best log2(gamma) is', + np.log2(best_gamma))
	print(gs_model.best_score_)
