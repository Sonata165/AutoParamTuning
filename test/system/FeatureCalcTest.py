'''
Author: 欧龙燊
本文件下函数都和数据集的特征计算有关
'''

import sys
import unittest

import pandas as pd

# import FeatureCalc
# from FeatureCalc import *
import knowledge.NetworkPrepare

sys.path.append('knowledge')




class TestFeatureCalc(unittest.TestCase):
	''' 测试FeaureCalc模块 '''
	def setUp(self):
		print("Do something beforehand")
		dataset = pd.read_csv('system/input/iris1.csv', sep=',', skipinitialspace=True)
		os.getcwd()

	def test_calculate_size(self):
		''' 测试calculate_size函数 '''
		self.assertEqual(5, calculate_attribute_num(dataset))

	# def


if __name__ == '__main__':
	unittest.main()
