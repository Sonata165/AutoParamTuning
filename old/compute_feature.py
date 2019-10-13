"""
本模块计算数据集特征：标准差、偏度
将数据保存在./nn_model/feature.csv中
输入：
SAVE_TO 保存结果的位置
DATASET_PATH 数据集路径
输出：
./nn_model/feature.csv
"""

import numpy as np
import pandas as pd
import _pickle as pk

SAVE_TO = './nn_model/feature.csv'
DISCRETE_BOUND = 10	# 当一个属性的取值小于等于它时，该属性被视作离散的
TOT_CORR_BOUND = 10000 # 当总相关性大于它时，将被记作它

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

# 开始干正事
SymPr = []	# 离散属性比率
Attr = []	# 属性个数
Obs = []	# 数据集大小
Entropy = []	# 信息熵
MultiInf = []	# 联合信息熵
Totcorr = []		# 交互信息
for data_path in DATASET_PATH:
	print(data_path)

	# 读入数据
	dataset = pd.read_csv(data_path,
		na_values='?', comment='\t', sep=',', skipinitialspace=True)
	print(type(dataset), len(dataset))
	dataset = dataset.dropna()
	raw_dataset = dataset.copy()
	label_name = raw_dataset.columns.values.tolist()[-1]
	y_train = raw_dataset.pop(label_name)
	X_train = raw_dataset.copy()

	# 数据集概况
	stat = dataset.describe().transpose()
	# print(stat)

	# feature1 计算离散属性比率
	discrete_count = 0
	for i in X_train.columns:
		if len(X_train[i].value_counts()) <= DISCRETE_BOUND:
			discrete_count += 1
	SymPr.append(discrete_count / X_train.columns.size)
	print(discrete_count, X_train.columns.size)

	# feature2 属性个数
	Attr.append(X_train.columns.size)

	# feature3 数据集大小
	n = len(dataset)
	Obs.append(n)

	# feature4 信息熵 -Sigma(P(yi)log_b(P(yi)))
	entropy = 0
	dic = y_train.value_counts()		# 它保存着不同标签的出现次数
	for i in dic:
		pr = i / n
		entropy -= pr * np.log2(pr)
	Entropy.append(entropy)

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
	MultiInf.append(multi_inf)

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
	Totcorr.append(min(tot_corr, TOT_CORR_BOUND))


	print()


d = {
	'Name': DATASET_PATH,
	'SymPr': SymPr,
	'Attr': Attr,
	'Obs': Obs,
	'Entropy': Entropy,
	'MultiInf': MultiInf,
	'Totcorr': Totcorr
}
d = pd.DataFrame(d)
d.to_csv(SAVE_TO, index=False)