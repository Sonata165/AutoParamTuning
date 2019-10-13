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

EPOCH = 100
DISCRETE_BOUND = 10
TOT_CORR_BOUND = 10000
C_MODEL_PATH = './nn_model/model_c.h5'
GAMMA_MODEL_PATH = './nn_model/model_gamma.h5'
LABEL_PATH = './nn_model/label.csv'
SAVE_TO = './nn_model/result3.csv'
DATASET_PATH = [
	# # 1
	# './BinaryClassificationDatabase/acute_inflammations1.csv',
	# './BinaryClassificationDatabase/acute_inflammations2.csv',
	# './BinaryClassificationDatabase/adult_sampling.csv',
	# './BinaryClassificationDatabase/audit.csv',
	# './BinaryClassificationDatabase/autism.csv',
	# './BinaryClassificationDatabase/autism-adolescent.csv',
	# './BinaryClassificationDatabase/autism-child.csv',
	# './BinaryClassificationDatabase/bank_modified_sampling.csv',
	# './BinaryClassificationDatabase/blood_transfusion_service_center.csv',
	# './BinaryClassificationDatabase/caesarian.csv',
	# './BinaryClassificationDatabase/chronic_kidney_disease.csv',
	# # 2
	# './BinaryClassificationDatabase/cmc_modified1.csv',
	# './BinaryClassificationDatabase/cmc_modified2.csv',
	# './BinaryClassificationDatabase/connect-4_modified1_sampling.csv',
	# './BinaryClassificationDatabase/connect-4_modified2_sampling.csv',
	# './BinaryClassificationDatabase/cryotherapy.csv',
	# './BinaryClassificationDatabase/haberman.csv',
	# './BinaryClassificationDatabase/immunotherapy.csv',
	# './BinaryClassificationDatabase/iris_modified1.csv',
	# './BinaryClassificationDatabase/iris_modified2.csv',
	# './BinaryClassificationDatabase/istanbul.csv',
	# './BinaryClassificationDatabase/mammographic.csv',
	# 3
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
	# # 4
	# './BinaryClassificationDatabase1/spectf.csv',
	# './BinaryClassificationDatabase1/thoracic.csv',
	# './BinaryClassificationDatabase1/wine.csv',

	# './BinaryClassificationDatabase2/breast-cancer.csv',
	# './BinaryClassificationDatabase2/breast-w.csv',
	# './BinaryClassificationDatabase2/clean1.csv',
	# './BinaryClassificationDatabase2/cmc_0_1.csv',
	# './BinaryClassificationDatabase2/cmc_0_2.csv',
	# './BinaryClassificationDatabase2/cmc_1_2.csv',
	# './BinaryClassificationDatabase2/credit6000_126.csv',
	# './BinaryClassificationDatabase2/credit-a.csv',
	# # 5
	# './BinaryClassificationDatabase2/credit-g.csv',
	# './BinaryClassificationDatabase2/cylinder-bands.csv',
	# './BinaryClassificationDatabase2/diabetes.csv',
	# './BinaryClassificationDatabase2/heart-statlog.csv',
	# './BinaryClassificationDatabase2/hepatitis.csv',
	# './BinaryClassificationDatabase2/ionosphere.csv',
	# './BinaryClassificationDatabase2/kr-vs-kp.csv',
	# './BinaryClassificationDatabase2/mushroom.csv',
	# './BinaryClassificationDatabase2/sonar.csv',
	# './BinaryClassificationDatabase2/spambase.csv',
	# './BinaryClassificationDatabase2/tic-tac-toe.csv',
	# './BinaryClassificationDatabase2/waveform-5000_0_1.csv',
	# './BinaryClassificationDatabase2/waveform-5000_0_2.csv',
	# './BinaryClassificationDatabase2/waveform-5000_1_2.csv'
]


C_nn = []
gamma_nn = []
score_nn = []
C_final = []
gamma_final = []
score_final = []
time_cost = []
for dataset in DATASET_PATH:
	print(dataset, ": ")

	t1 = time.time()

	''' 数据预处理 '''
	# 读入数据
	raw_dataset = pd.read_csv(dataset,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)
	raw_dataset = raw_dataset.dropna()
	dataset = raw_dataset.copy()
	label_name = raw_dataset.columns.values.tolist()[-1]
	y_train = raw_dataset.pop(label_name)
	X_train = raw_dataset.copy()

	label = pd.read_csv(LABEL_PATH,
		na_values='?', comment='\t', sep=',', skipinitialspace=True)


	''' 特征计算 '''
	# feature1 计算离散属性比率
	discrete_count = 0
	for i in X_train.columns:
		if len(X_train[i].value_counts()) <= DISCRETE_BOUND:
			discrete_count += 1
	SymPr = discrete_count / X_train.columns.size

	# feature2 属性个数
	Attr = X_train.columns.size

	# feature3 数据集大小
	n = len(dataset)
	Obs = n

	# feature4 信息熵 -Sigma(P(yi)log_b(P(yi)))
	entropy = 0
	dic = y_train.value_counts()		# 它保存着不同标签的出现次数
	for i in dic:
		pr = i / n
		entropy -= pr * np.log2(pr)
	Entropy = entropy

	# feature5 联合信息熵 -Sigma(P(x1,x2)log_b(P(x1,x2)))
	lines = []
	for index, row in dataset.iterrows():
		line_list = list(row)
		lines.append(str(line_list))
	lines = pd.Series(lines)
	dic = lines.value_counts()		# 它保存着每行的toString和不同行的出现次数
	b = len(dic)					# b表示有多少种不同的行
	multi_inf = 0
	for i in dic:
		pr = i / n					# 该行出现的概率
		multi_inf -= pr * np.log2(pr)
	MultiInf = multi_inf

	# feature6 总相关性    Sigma( P(x1, x2) * log( P(x1,x2)/(P(x1)P(x2)) ) )
	# 建立一个<line, 次数>的dic
	dic = dict(dic)
	# 对每一列都建立一个字典
	dic_of_column = []
	for index in dataset.columns:
		dic_of_column.append(dict(dataset[index].value_counts()))
	# print(dic_of_column)
	# 对于每行，对于每个元素，计算概率乘积，然后按规则计算
	tot_corr = 0
	for line_str in dic:
		line_tmp = line_str[1:-1].split(', ')
		line = []
		for i in line_tmp:
			line.append(np.float64(i))
		# print(line)
		product = np.float64(1.0)
		# print(line_str)
		for i in range(0, len(line)):
			# print(type(dic_of_column[i][line[i]]))
			product *= dic_of_column[i][line[i]] / n
		# print(product)			# 该行各元素概率乘积
		pr = dic[line_str] / n 		# 该行出现的概率
		tot_corr += pr * np.log(pr/product)
		# print(tot_corr)
	# print(tot_corr)
	Totcorr = min(tot_corr, TOT_CORR_BOUND)



	feature = [SymPr, Attr, Obs, Entropy, MultiInf, Totcorr]
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

	print('神经网络结果')
	print("Predicted C is", predicted_C)
	print("Predicted gamma is", predicted_gamma)
	model = SVC(C=predicted_C, kernel='rbf', gamma=predicted_gamma)
	scores = cross_val_score(model, X_train, y_train, cv=5)
	print("95/100置信区间Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

	C_nn.append(log2_C)
	gamma_nn.append(log2_gamma)
	score_nn.append(scores.mean())

	''' 找局部最优 '''
	for i in range(0, EPOCH+1):
		log2_C, log2_gamma = cbz_method.optimize(X_train, y_train, log2_C, log2_gamma)

	predicted_C = 2**log2_C
	predicted_gamma = 2**log2_gamma
	print('最终结果')
	print("Finally, Predicted C is", predicted_C)
	print("Finally, Predicted gamma is", predicted_gamma)
	model = SVC(C=predicted_C, kernel='rbf', gamma=predicted_gamma)
	scores = cross_val_score(model, X_train, y_train, cv=5)
	print("95/100置信区间Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
	print()

	C_final.append(log2_C)
	gamma_final.append(log2_gamma)
	score_final.append(scores.mean())

	t3 = time.time()

	time_cost.append(t3-t2)


d = pd.DataFrame({
	'Name': DATASET_PATH,
	'C_nn': C_nn,
	'gamma_nn': gamma_nn,
	'score_nn': score_nn,
	'C_final': C_final,
	'gamma_final': gamma_final,
	'score_final': score_final,
	'Time': time_cost
})
d.to_csv(SAVE_TO, index=False)




