'''
Author: 欧龙燊
本文件下函数都和数据集的特征计算有关
'''

import unittest
import sys
import pandas as pd
import os
sys.path.append('../')

from FeatureCalc import *


class TestFeatureCalc(unittest.TestCase):
	''' 测试FeaureCalc模块 '''
	def setUp(self):
		print("Do something beforehand")
		self.dataset = pd.read_csv('system/input/iris1.csv', sep=',', skipinitialspace=True)
		os.getcwd()

	def test_calculate_size(self):
		''' 测试calculate_size函数 '''
		self.assertEqual(5, calculate_attribute_num(self.dataset))

	# def


if __name__ == '__main__':
	unittest.main()