'''
Author: 欧龙燊
本文件下函数都和数据集的特征计算有关
'''

import numpy as np
import pandas as pd

DISCRETE_BOUND = 10	# 当一个属性的取值小于等于它时，该属性被视作离散的
TOT_CORR_BOUND = 10000 # 当总相关性大于它时，将被记作它
FEATURENAMES = [
	'attribute_num',
	'discrete_ratio',
	'entropy',
	'joint_inf',
	'size',
	'totcorr'
]

def calculate_features(dataset):
	'''
	计算特征
	Parameters:
	  dataset - 一个待计算特征的数据集
	Returns:
	  列表，所有计算好的特征
	'''
	attribute_num = calculate_attribute_num(dataset)
	discrete_ratio = calculate_discrete_ratio(dataset)
	entropy = calculate_entropy(dataset)
	joint_inf = calculate_joint_inf(dataset)
	size = calculate_size(dataset)
	totcorr = calculate_totcorr(dataset)

	ret = [attribute_num, discrete_ratio, entropy, joint_inf, size, totcorr]
	return ret


def get_feature_name():
	'''
	获取计算过的特征的名称
	'''
	return FEATURENAMES


def calculate_size(dataset):
	'''
	计算数据集大小
	Parameters:
	  dataset - 待计算的数据集
	Returns:
	  该数据集的大小（样本数目）
	'''
	return len(dataset)


def calculate_discrete_ratio(dataset):
	"""
	计算数据集的离散属性比率
	Parameters:
	  dataset - 待计算的数据集
	Returns:
	  该数据集的离散属性比率
	"""
	discrete_count = 0
	for i in dataset.columns:
		if len(dataset[i].value_counts()) == 2:
			discrete_count += 1
	return discrete_count / dataset.columns.size


def calculate_attribute_num(dataset):
	"""
	计算数据集属性个数
	Parameters:
	  dataset - 待计算的数据集
	Returns:
	  该数据集的属性个数
	"""
	return dataset.columns.size


def calculate_entropy(dataset):
	"""
	计算指定数据集的信息熵(只看标签一栏) -Sigma(P(yi)log_2(P(yi)))  加和次数等于标签数目
	Parameters:
	  dataset - 待计算的数据集
	Returns:
	  该数据集的信息熵
	"""
	entropy = 0
	dic = dataset['Label'].value_counts()		# 它保存着不同标签的出现次数
	n = len(dataset)
	for i in dic:
		pr = i / n
		entropy -= pr * np.log2(pr)
	return entropy


def calculate_joint_inf(dataset):
	"""
	计算指定数据集的联合信息熵 -Sigma(P(x1,x2)log_2(P(x1,x2)))
	TODO Buggy
	Parameters:
	  dataset - 待计算的数据集
	Returns:
	  该数据集的联合信息熵
	"""
	lines = []
	for index, row in dataset.iterrows():
		line_list = list(row)
		lines.append(str(line_list))
	lines = pd.Series(lines)
	dic = lines.value_counts()		# 它保存着每行的toString和不同行的出现次数
	b = len(dic)					# b表示有多少种不同的行
	joint_inf = 0
	n = len(dataset)
	for i in dic:
		pr = i / n					# 该行出现的概率
		joint_inf -= pr * np.log2(pr)
	return joint_inf





def calculate_totcorr(dataset):
	"""
	对一个数据集计算总相关性
	TODO Buggy
	Parameters:
	  dataset - 待计算的数据集
	Returns:
	  该数据集的总相关性
	"""
	lines = []
	for index, row in dataset.iterrows():
		line_list = list(row)
		lines.append(str(line_list))
	lines = pd.Series(lines)
	dic = lines.value_counts()		# 它保存着每行的toString和不同行的出现次数
	# 建立一个<line, 次数>的dic
	dic = dict(dic)
	# 对每一列都建立一个字典
	dic_of_column = []
	for index in dataset.columns:
		dic_of_column.append(dict(dataset[index].value_counts()))
	# 对于每行，对于每个元素，计算概率乘积，然后按规则计算
	tot_corr = 0
	for line_str in dic:
		line_tmp = line_str[1:-1].split(', ')
		line = []
		for i in line_tmp:
			line.append(np.float64(i))
		product = np.float64(1.0)
		n = len(dataset)
		for i in range(0, len(line)):
			product *= dic_of_column[i][line[i]] / n
		pr = dic[line_str] / n 		# 该行出现的概率
		tot_corr += pr * np.log(pr/product)
	return min(tot_corr, TOT_CORR_BOUND)


def main():
	print("")


if __name__ == '__main__':
	main()