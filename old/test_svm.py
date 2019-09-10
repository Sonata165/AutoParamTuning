"""
该模块对给定的数据集使用SVM进行分类，给出95%置信区间的交叉验证正确率评估
目的是用来检测数据集格式是否正确
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
GAMMA = 2

# 数据集路径
# DATASET_PATH = [
# 	'./Preconditioned Data/Binary Classification Database/acute_inflammations1.csv',
# 	'./Preconditioned Data/Binary Classification Database/acute_inflammations2.csv',
# 	'./Preconditioned Data/Binary Classification Database/adult_sampling.csv',
# 	'./Preconditioned Data/Binary Classification Database/audit.csv',
# 	'./Preconditioned Data/Binary Classification Database/autism.csv',
# 	'./Preconditioned Data/Binary Classification Database/autism-adolescent.csv',
# 	'./Preconditioned Data/Binary Classification Database/autism-child.csv',
# 	'./Preconditioned Data/Binary Classification Database/bank_modified_sampling.csv',
# 	'./Preconditioned Data/Binary Classification Database/blood_transfusion_service_center.csv',
# 	'./Preconditioned Data/Binary Classification Database/caesarian.csv',
# 	'./Preconditioned Data/Binary Classification Database/chronic_kidney_disease.csv',
# 	'./Preconditioned Data/Binary Classification Database/cmc_modified1.csv',
# 	'./Preconditioned Data/Binary Classification Database/cmc_modified2.csv',
# 	'./Preconditioned Data/Binary Classification Database/connect-4_modified1_sampling.csv',
# 	'./Preconditioned Data/Binary Classification Database/connect-4_modified2_sampling.csv',
# 	'./Preconditioned Data/Binary Classification Database/cryotherapy.csv',
# 	'./Preconditioned Data/Binary Classification Database/haberman.csv',
# 	'./Preconditioned Data/Binary Classification Database/immunotherapy.csv',
# 	'./Preconditioned Data/Binary Classification Database/iris_modified1.csv',
# 	'./Preconditioned Data/Binary Classification Database/iris_modified2.csv',
# 	'./Preconditioned Data/Binary Classification Database/istanbul.csv',
# 	'./Preconditioned Data/Binary Classification Database/mammographic.csv',
# 	'./Preconditioned Data/Binary Classification Database/seeds_modified.csv',
# 	'./Preconditioned Data/Binary Classification Database/somerville.csv',
# 	'./Preconditioned Data/Binary Classification Database/vertebral.csv',
# ]
# DATASET_PATH = [
# 	'./BinaryClassificationDatabase1/breast.csv',
# 	'./BinaryClassificationDatabase1/breast-cancer(diagnostic).csv',
# 	'./BinaryClassificationDatabase1/breast-cancer.csv',
# 	'./BinaryClassificationDatabase1/climate-model.csv',
# 	'./BinaryClassificationDatabase1/connectionist.csv',
# 	'./BinaryClassificationDatabase1/extention.csv',
# 	'./BinaryClassificationDatabase1/fertility.csv',
# 	'./BinaryClassificationDatabase1/ilpd.csv',
# 	'./BinaryClassificationDatabase1/monk (1).csv',
# 	'./BinaryClassificationDatabase1/monk (2).csv',
# 	'./BinaryClassificationDatabase1/monk (3).csv',
# 	'./BinaryClassificationDatabase1/spectf.csv',
# 	'./BinaryClassificationDatabase1/thoracic.csv',
# 	'./BinaryClassificationDatabase1/wine.csv'
# ]
DATASET_PATH = [
	'BinaryClassificationDatabase2/breast-cancer.csv',
	'BinaryClassificationDatabase2/breast-w.csv',
	'BinaryClassificationDatabase2/clean1.csv',
	'BinaryClassificationDatabase2/cmc_0_1.csv',
	'BinaryClassificationDatabase2/cmc_0_2.csv',
	'BinaryClassificationDatabase2/cmc_1_2.csv',
	'BinaryClassificationDatabase2/credit6000_126.csv',
	'BinaryClassificationDatabase2/credit-a.csv',
	'BinaryClassificationDatabase2/credit-g.csv',
	'BinaryClassificationDatabase2/cylinder-bands.csv',
	'BinaryClassificationDatabase2/diabetes.csv',
	'BinaryClassificationDatabase2/heart-statlog.csv',
	'BinaryClassificationDatabase2/hepatitis.csv',
	'BinaryClassificationDatabase2/ionosphere.csv',
	'BinaryClassificationDatabase2/kr-vs-kp.csv',
	'BinaryClassificationDatabase2/mushroom.csv',
	'BinaryClassificationDatabase2/sonar.csv',
	'BinaryClassificationDatabase2/spambase.csv',
	'BinaryClassificationDatabase2/tic-tac-toe.csv',
	'BinaryClassificationDatabase2/waveform-5000_0_1.csv',
	'BinaryClassificationDatabase2/waveform-5000_0_2.csv',
	'BinaryClassificationDatabase2/waveform-5000_1_2.csv'
]

for data_path in DATASET_PATH:
	# 读入数据
	print(data_path)
	raw_dataset = pd.read_csv(data_path,
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



# # 多标准评估
# scoring = ['precision_macro', 'recall_macro']
# scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring,
# 	return_train_score=False)
# print(scores)
