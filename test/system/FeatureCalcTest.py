'''
Author: 欧龙燊
测试特征计算文件
'''

import sys
import unittest
import pandas as pd

sys.path.append('knowledge')
sys.path.append('system')

from FeatureCalc import *

class TestFeatureCalc(unittest.TestCase):
	''' 测试FeaureCalc模块 '''

	def test_calculate_size(self):
		''' 测试calculate_size函数 '''
		dataset1 = read_dataset('test/system/input/iris1.csv')
		self.assertEqual(100, calculate_size(dataset1))
		dataset2 = read_dataset('test/system/input/LongIris.csv')
		self.assertEqual(200, calculate_size(dataset2))

	def test_calculate_discrete_ratio(self):
		''' 测试离散属性比率计算 '''
		dataset1 = read_dataset('test/system/input/Simple1.csv')
		self.assertEqual(0.4, calculate_discrete_ratio(dataset1))
		dataset2 = read_dataset('test/system/input/Simple2.csv')
		self.assertEqual(0.6, calculate_discrete_ratio(dataset2))

	def test_calculate_attribute_num(self):
		dataset1 = read_dataset('test/system/input/iris1.csv')
		self.assertEqual(5, calculate_attribute_num(dataset1))
		dataset2 = read_dataset('test/system/input/DoubleIris.csv')
		self.assertEqual(10, calculate_attribute_num(dataset2))

	def test_calculate_entropy(self):
		# 一个20行常规数据集，两类
		dataset = read_dataset('test/system/input/Simple1.csv')
		ans = calculate_entropy(dataset) # should be 0.8112
		self.assertTrue(0.8111 < ans and ans < 0.8113)

		# 一个20行常规数据集，两类
		dataset = read_dataset('test/system/input/Simple3.csv')
		ans = calculate_entropy(dataset) # should be 2.3219
		self.assertTrue(2.3218 < ans and ans < 2.3220)

		# 10000行，10000类
		dataset = read_dataset('test/system/input/Long.csv')
		ans = calculate_entropy(dataset) # 13.2877
		self.assertTrue(13.2876 < ans and ans < 13.2878)

	def test_calculate_joint_inf(self):
		dataset = read_dataset('test/system/input/Simple4.csv')
		ans = calculate_joint_inf(dataset) # 2.5219
		self.assertTrue(2.5218 < ans and ans < 2.5220)

	def



def read_dataset(PATH):
	'''
	按照约定方法读入数据集
	'''
	dataset = pd.read_csv(PATH, sep=',', skipinitialspace=True)
	return dataset


if __name__ == '__main__':
	unittest.main()
