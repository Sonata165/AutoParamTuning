"""
4.12未使用该模块

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
REDUCE_TO = 3

# 数据集路径
DATASET_PATH = [
	'BinaryClassificationDatabase/acute_inflammations1.csv',
	'BinaryClassificationDatabase/acute_inflammations2.csv',
	'BinaryClassificationDatabase/adult_sampling.csv',
	'BinaryClassificationDatabase/audit.csv',
	'BinaryClassificationDatabase/autism.csv',
	'BinaryClassificationDatabase/autism-adolescent.csv',
	'BinaryClassificationDatabase/autism-child.csv',
	'BinaryClassificationDatabase/bank_modified_sampling.csv',
	'BinaryClassificationDatabase/blood_transfusion_service_center.csv',
	'BinaryClassificationDatabase/caesarian.csv',
	'BinaryClassificationDatabase/chronic_kidney_disease.csv',
	'BinaryClassificationDatabase/cmc_modified1.csv',
	'BinaryClassificationDatabase/cmc_modified2.csv',
	'BinaryClassificationDatabase/connect-4_modified1_sampling.csv',
	'BinaryClassificationDatabase/connect-4_modified2_sampling.csv',
	'BinaryClassificationDatabase/cryotherapy.csv',
	'BinaryClassificationDatabase/haberman.csv',
	'BinaryClassificationDatabase/immunotherapy.csv',
	'BinaryClassificationDatabase/iris_modified1.csv',
	'BinaryClassificationDatabase/iris_modified2.csv',
	'BinaryClassificationDatabase/istanbul.csv',
	'BinaryClassificationDatabase/mammographic.csv',
	'BinaryClassificationDatabase/seeds_modified.csv',
	'BinaryClassificationDatabase/somerville.csv',
	'BinaryClassificationDatabase/vertebral.csv',

	'BinaryClassificationDatabase1/breast.csv',
	'BinaryClassificationDatabase1/breast-cancer(diagnostic).csv',
	'BinaryClassificationDatabase1/climate-model.csv',
	'BinaryClassificationDatabase1/connectionist.csv',
	'BinaryClassificationDatabase1/extention.csv',
	'BinaryClassificationDatabase1/fertility.csv',
	'BinaryClassificationDatabase1/ilpd.csv',
	'BinaryClassificationDatabase1/monk (1).csv',
	'BinaryClassificationDatabase1/spectf.csv',
	'BinaryClassificationDatabase1/thoracic.csv',
	'BinaryClassificationDatabase1/wine.csv',

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

# 开始干正事
for data_path in DATASET_PATH:
	print(data_path)
	# 读入数据
	dataset = pd.read_csv(data_path,
		na_values='?', comment='\t', sep=',', skipinitialspace=True)

	# 丢弃缺失数据
	dataset = dataset.dropna()

	# 抽样
	# sample_num = min(len(dataset), 150)
	# dataset = dataset.sample(n=sample_num)

	dataset.to_csv('./PreconditionedData/' + data_path, index=False)
	print()
