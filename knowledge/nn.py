"""
一个临时模块，用作测试
创建一个神经网络，保存到network目录下
输入维度和FeatureCalc.get_feature_name返回值长度一致，输出维度为1
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append('system')

import FeatureCalc

FEATURE_PATH = './nn_model/feature.csv'
LABEL_PATH = './nn_model/label.csv'
EPOCHS = 2000

input_dimension = len(FeatureCalc.get_feature_name('SVM'))

# 建立模型
def build_model():
	model = keras.Sequential([
		layers.Dense(4, activation=tf.nn.sigmoid, input_shape=[input_dimension]),
		layers.Dense(16, activation=tf.nn.sigmoid),
		layers.Dense(1)
	])
	model.compile(
		loss = 'mse',
		optimizer = tf.keras.optimizers.RMSprop(0.001),		# 学习率初始值为0.001
		metrics = ['mae', 'mse']							# 评估指标: [平均绝对误差, 均方误差]
	)
	return model

model = build_model()

''' 保存网络 '''
model.save('system/network/SvmModel.h5')
print('模型已保存')
