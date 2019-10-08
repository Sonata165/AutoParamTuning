'''
计算database目录下所有数据集的特征和最优参数（网格搜索）
database目录结构如README中所示
计算特征时，可以调用system/FeatureCalc.py中的calculate_features(dataset)，
其中dataset是一个读入的数据集（用read_csv函数读入，类型为DataFrame），返回值是所有特征的列表
计算最优参数时，可以参考原来的old/compute_label.py
全部计算完后，将结果保存在如README所示knowledge下三个csv文件中
保存格式如下：
	需要把最优参数放在后面的列，特征放在前面的列
	第一行是特征名和参数名，从第二行开始是数据
	第一列不是编号，从第一列开始就是数据
'''

def get_param_name(alg_name):
	'''
	返回待调的超参数的名称
	Parameters:
	  alg_name - String类型，待调参算法名称，'SVM','ElasticNet','GMM'中的一个
	Returns:
	  一个列表，包含该算法要调的参数的名称。
	  要求列表大小和神经网络输出数目总数保持一致，顺序也和神经网络输出保持一致
	'''
	return ['a']

 -*- coding: utf-8 -*-
