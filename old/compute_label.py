"""
此模块对数据集使用网格搜索找到最优参数
输入：
GRID_STRIDE 网格搜索步长（指数）
UPPER_BOUND 网格搜索上界
LOWER_BOUND 网格搜索下界
DATASET_PATH 数据集路径
CV 交叉验证折数
SAVE_TO 保存结果的位置
输出：
./nn_model/label.csv 该文件包含两列，第一列为log2(C)，第二列为log2(gamma)
"""
print('开始导入库...')
import time
import numpy as np
import pandas as pd
import _pickle as pk
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.datasets import load_digits
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
print('导入成功！')

GRID_STRIDE = 1
UPPER_BOUND = 10
LOWER_BOUND = -10
CV = 5
SAVE_TO = './nn_model/label.csv'

# 数据集路径
DATASET_PATH = [
	'./BinaryClassificationDatabase/acute_inflammations1.csv',
	'./BinaryClassificationDatabase/acute_inflammations2.csv',
	'./BinaryClassificationDatabase/adult_sampling.csv',
	'./BinaryClassificationDatabase/audit.csv',
	'./BinaryClassificationDatabase/autism.csv',
	'./BinaryClassificationDatabase/autism-adolescent.csv',
	'./BinaryClassificationDatabase/autism-child.csv',
	'./BinaryClassificationDatabase/bank_modified_sampling.csv',
	'./BinaryClassificationDatabase/blood_transfusion_service_center.csv',
	'./BinaryClassificationDatabase/caesarian.csv',
	'./BinaryClassificationDatabase/chronic_kidney_disease.csv',
	'./BinaryClassificationDatabase/cmc_modified1.csv',
	'./BinaryClassificationDatabase/cmc_modified2.csv',
	'./BinaryClassificationDatabase/connect-4_modified1_sampling.csv',
	'./BinaryClassificationDatabase/connect-4_modified2_sampling.csv',
	'./BinaryClassificationDatabase/cryotherapy.csv',
	'./BinaryClassificationDatabase/haberman.csv',
	'./BinaryClassificationDatabase/immunotherapy.csv',
	'./BinaryClassificationDatabase/iris_modified1.csv',
	'./BinaryClassificationDatabase/iris_modified2.csv',
	'./BinaryClassificationDatabase/istanbul.csv',
	'./BinaryClassificationDatabase/mammographic.csv',
	'./BinaryClassificationDatabase/seeds_modified.csv',
	'./BinaryClassificationDatabase/somerville.csv',
	'./BinaryClassificationDatabase/vertebral.csv',

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

	'./BinaryClassificationDatabase2/breast-cancer.csv',
	'./BinaryClassificationDatabase2/breast-w.csv',
	'./BinaryClassificationDatabase2/clean1.csv',
	'./BinaryClassificationDatabase2/cmc_0_1.csv',
	'./BinaryClassificationDatabase2/cmc_0_2.csv',
	'./BinaryClassificationDatabase2/cmc_1_2.csv',
	'./BinaryClassificationDatabase2/credit6000_126.csv',
	'./BinaryClassificationDatabase2/credit-a.csv',
	'./BinaryClassificationDatabase2/credit-g.csv',
	'./BinaryClassificationDatabase2/cylinder-bands.csv',
	'./BinaryClassificationDatabase2/diabetes.csv',
	'./BinaryClassificationDatabase2/heart-statlog.csv',
	'./BinaryClassificationDatabase2/hepatitis.csv',
	'./BinaryClassificationDatabase2/ionosphere.csv',
	'./BinaryClassificationDatabase2/kr-vs-kp.csv',
	'./BinaryClassificationDatabase2/mushroom.csv',
	'./BinaryClassificationDatabase2/sonar.csv',
	'./BinaryClassificationDatabase2/spambase.csv',
	'./BinaryClassificationDatabase2/tic-tac-toe.csv',
	'./BinaryClassificationDatabase2/waveform-5000_0_1.csv',
	'./BinaryClassificationDatabase2/waveform-5000_0_2.csv',
	'./BinaryClassificationDatabase2/waveform-5000_1_2.csv'
]

# 网格搜索相关数据
big_array = []
i = LOWER_BOUND
while i <= UPPER_BOUND + 0.001:
	big_array.append(2**i)
	i += GRID_STRIDE


# 开始做正事
C = []
gamma = []
score = []
gs_time = []
for data_path in DATASET_PATH:

	t1 = time.time()

	# 读入数据
	print(data_path)
	raw_dataset = pd.read_csv(data_path,
		na_values='?', comment='\t', sep=',', skipinitialspace=True)
	print(type(raw_dataset), len(raw_dataset))
	raw_dataset = raw_dataset.dropna()
	label_name = raw_dataset.columns.values.tolist()[-1]
	y_train = raw_dataset.pop(label_name)
	X_train = raw_dataset.copy()

	# 转换格式来加速
	X_train = np.array(X_train)
	y_train = np.array(y_train)

	# 训练
	print(big_array)
	param_grid = [{
		'C': big_array,
		'gamma': big_array,
		'kernel': ['rbf']
	}]
	model = SVC()
	gs_model = GridSearchCV(model, param_grid, cv=CV, iid=True)
	gs_model.fit(X_train, y_train)

	t2 = time.time()

	# 展示结果
	print(gs_model.best_params_)
	print('best score is ', gs_model.best_score_)
	C.append(gs_model.best_params_['C'])
	gamma.append(gs_model.best_params_['gamma'])
	score.append(gs_model.best_score_)
	gs_time.append(t2-t1)

	print()


C = np.log2(C)
gamma = np.log2(gamma)
data = pd.DataFrame({
	'C':C,
	'gamma':gamma,
	'score':score,
	'time':gs_time
})
print(data)

data.to_csv(SAVE_TO, index=False)
