"""
该模块创建神经网络拟合./nn_model/feature.csv与label.csv中的数据，
输出拟合结果，并保存模型至./nn_model/目录下
输入：
FEATURE_PATH 数据集的特征组成的训练集的属性
LABEL_PATH 网格搜索的最优参数，为训练集标签
EPOCHS 训练轮数
SAVE_C_MODEL_TO 保存C的模型的路径
SAVE_GAMMA_MODEL_TO 保存GAMMA的模型的路径
输出：
C的模型
gamma的模型
"""

print('开始导入库...')
import _pickle as pickle
import os
import pandas as pd
import seaborn as sb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers

print('导入成功！')

FEATURE_PATH = './nn_model/feature.csv'
LABEL_PATH = './nn_model/label.csv'
EPOCHS = 2000
SAVE_C_MODEL_TO = './nn_model/model_c.h5'
SAVE_GAMMA_MODEL_TO = './nn_model/model_gamma.h5'
"""
# 读入数据
X_train = pd.read_csv(FEATURE_PATH,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)
X_train.pop('Name')
# print(type(X_train), len(X_train))
y_train = pd.read_csv(LABEL_PATH,
	na_values='?', comment='\t', sep=',', skipinitialspace=True)
# print(type(y_train), len(y_train))
# print(y_train)
C = y_train['C']
gamma = y_train['gamma']
"""
from knowledge.NetworkPrepare import *
import os

os.chdir("../../")
X_train, y_train = read_data("SVM")
C = y_train[:, 0]
gamma = y_train[:, 1]
# # 看一眼数据集
# print(X_train.describe())
# sb.pairplot(X_train[['std0','std1','skew0','skew1']]
# 	,diag_kind = 'kde')
# print(X_train)
# plt.show()

''' 使用标准化的数据集训练后，我不太明白如何用未标准化的数据测试，所以暂时没有标准化 '''
# # 标准化训练集
# def norm(x):
# 	return (x - train_stats['mean']) / train_stats['std']

# normed_train_data = norm(train_data)

# print('标准化的训练集：')
# print(normed_train_data)
# print('标签(C)：')
# print(train_labels_c)
# print('标签(gamma): ')
# print(train_labels_gamma)
# print(type(train_labels_c))


''' 训练神经网络 '''


# 建立模型
def build_model(loss):
    model = keras.Sequential([
        layers.Dense(36, input_shape=(36,),activity_regularizer=keras.regularizers.l1_l2(0.01,0.01)),
        layers.BatchNormalization(),
        layers.Activation('tanh'),
        layers.Dropout(0.2),
        layers.Dense(16, activity_regularizer=keras.regularizers.l1_l2(0.01,0.01)),
        layers.BatchNormalization(),
        layers.Activation('tanh'),
        layers.Dropout(0.2),
        layers.Dense(16, activity_regularizer=keras.regularizers.l1_l2(0.01,0.01)),
        layers.BatchNormalization(),
        layers.Activation('tanh'),
        layers.Dropout(0.2),
        layers.Dense(4, activity_regularizer=keras.regularizers.l1_l2(0.01, 0.01)),
        layers.BatchNormalization(),
        layers.Activation('tanh'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(0.01),  # 学习率初始值为0.001
        metrics=['mae', 'mse']  # 评估指标: [平均绝对误差, 均方误差]
    )
    return model
    # return reg_net((36,))

model_c = build_model('mse')
model_gamma = build_model('mae')
model_c.summary()


# 训练
class PrintDot(keras.callbacks.Callback):  # 一个回调函数
    def on_epoch_end(self, epoch, logs):
        rnd = epoch + 1
        if rnd % 5 == 0:
            print('.', end='')
        if rnd % 100 == 0:
            print('')


# 创建early_stop
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=300)  # 用EarlyStopping创建另一个回调函数
reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100)

print('开始训练c')
history_c = model_c.fit(
    X_train, C, epochs=EPOCHS,
    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot(),reduceLR]
)
print('训练完成！')

print('开始训练gamma')
history_gamma = model_gamma.fit(
    X_train, gamma, epochs=EPOCHS,
    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()]
)
print('训练完成！')


def plot_history_c(history):
    hist = pd.DataFrame(history.history)  # 将history从dict类型转换为DataFrame类型
    print(hist)
    hist['epoch'] = history.epoch  # 添加epoch列

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [C]')
    plt.plot(hist['epoch'], hist['mae'],  # 画平均绝对误差图
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],  # 画验证集平均绝对误差图
             label='Val Error')
    plt.legend()

    # plt.ylim([0,100])												# 设置y轴范围

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$C^2$]')
    plt.plot(hist['epoch'], hist['mse'],  # 画均方误差图
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],  # 画验证集均方误差图
             label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss [$C^2$]')
    plt.plot(hist['epoch'], hist['loss'],  # 画均方误差图
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],  # 画验证集均方误差图
             label='Val Error')
    plt.legend()


# plt.ylim([0,32000])											# 设置y轴范围

def plot_history_gamma(history):
    hist = pd.DataFrame(history.history)  # 将history从dict类型转换为DataFrame类型
    print(hist)
    hist['epoch'] = history.epoch  # 添加epoch列

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [gamma]')
    plt.plot(hist['epoch'], hist['mae'],  # 画平均绝对误差图
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],  # 画验证集平均绝对误差图
             label='Val Error')
    plt.legend()

    # plt.ylim([0,100])												# 设置y轴范围

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$gamma^2$]')
    plt.plot(hist['epoch'], hist['mse'],  # 画均方误差图
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],  # 画验证集均方误差图
             label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss [$gamma^2$]')
    plt.plot(hist['epoch'], hist['loss'],  # 画均方误差图
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],  # 画验证集均方误差图
             label='Val Error')
    plt.legend()


# 添加图例
# plt.ylim([0,32000])											# 设置y轴范围

plot_history_c(history_c)
plt.show()

plot_history_gamma(history_gamma)
plt.show()

''' 保存网络 '''
"""
model_c.save(SAVE_C_MODEL_TO)
model_gamma.save(SAVE_GAMMA_MODEL_TO)
print('模型已保存')
"""
