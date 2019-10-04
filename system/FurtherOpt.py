
def find_local_supreme(dataset, alg_name, predicted_param):
	'''
	在神经网络给出结果的基础上，寻找局部极值，进一步优化超参数.
	Parameters:
	  dataset - 一个用户数据集，类型为pandas.DataFrame，每行一个样本，第一行是属性名称，第二行起是数据
	  alg_name - 待调参算法名称，String类型，'SVM','ElasticNet','GMM'中的一个
	  predicted_param - 神经网络的预测结果，一个列表，包含各个参数预测的值
	Returns:
	  一个列表，包含各个参数的最终优化结果，要求参数次序和predicted_param相同
	'''
	# return predicted_param
	return [0.5]