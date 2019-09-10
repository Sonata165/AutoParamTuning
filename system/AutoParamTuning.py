'''
客户端
前提：用户的数据集已经转化为数值类型，但还未标准化
'''

import pandas as pd
from tensorflow import keras

import FeatureCalc
import FurtherOpt

def main():
	print("选择算法")
	print("1. SVM分类")
	print("2. 弹性网络回归")
	print("3. GMM聚类")
	choice = int(input("> "))
	if choice == 1:
		alg_name = 'SVM'
		model_path = "SvmModel.h5"
	elif choice == 2:
		alg_name = 'ElasticNet'
		model_path = "ElasticNetModel.h5"
	elif choice == 3:
		alg_name = 'GMM'
		model_path = "GmmModel.h5"

	dataset = read_dataset()
	features = FeatureCalc.calculate_features(dataset)

	model = keras.models.load_model('network/' + model_path)
	predicted_param = model.predict(features)
	print('神经网络预测结果为：', predicted_param)

	optimized_param = FurtherOpt.find_local_supreme(dataset, alg_name, predicted_param)
	print('进一步优化结果为：', optimized_param)


def read_dataset():
	print('读取数据集')
	dataset = pd.read_csv() # TODO
	return dataset

if __name__ == '__main__':
	main()