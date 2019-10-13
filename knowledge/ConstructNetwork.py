import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers

EPOCHS = 1000

def main():
    print('1. svm 2. elasticnet 3. gmm')
    choice = int(input('> '))
    if choice == 1:
        svm_nn()
    elif choice == 2:
        elasticnet_nn()
    elif choice == 3:
        gmm_nn()

def svm_nn():
    X = pd.read_csv('knowledge/SVM_features.csv')
    label = pd.read_csv('knowledge/SVM_labels.csv')
    y_c = label['C']
    y_gamma = label['gamma']

    regression_net('SVM_C', X, y_c)
    regression_net('SVM_gamma', X, y_gamma)
    print()

def elasticnet_nn():
    X = pd.read_csv('knowledge/ElasticNet_features.csv')
    label = pd.read_csv('knowledge/ElasticNet_labels.csv')
    y_alpha = label['alpha']
    y_l1 = label['l1_ratio']

    regression_net('ElasticNet_alpha', X, y_alpha)
    regression_net('ElasticNet_l1_ratio', X, y_l1)
    print()

def gmm_nn():
    X = pd.read_csv('knowledge/GMM_features.csv')
    label = pd.read_csv('knowledge/GMM_labels.csv')
    y_ct = label['covariance_type']
    y_nc = label['n_components']

    classification_net('GMM_covariance_type', X, y_ct, 4)
    regression_net('GMM_n_components', X, y_nc)
    print()

def classification_net(output_name, X, y, class_num):
    model = build_classification_model(X, class_num)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X, y, epochs=EPOCHS, validation_split=0.2, 
        verbose=0,  callbacks=[early_stop, PrintDot()]
    )
    plot_history(history)
    plt.show()
    model.save('system/network/' + output_name + '.h5')
    print()

def regression_net(output_name, X, y):
    model = build_regression_model(X)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        X, y, epochs=EPOCHS,
        validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()]
    )
    plot_history(history)
    plt.show()
    model.save('system/network/' + output_name + '.h5')

def build_classification_model(X, class_num):
    model = keras.Sequential([
        layers.Dense(64, input_shape=[len(X.keys())]),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(class_num, activation=keras.activations.softmax)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model
    print()

def build_regression_model(X):
	model = keras.Sequential([
		layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X.keys())]),
		layers.Dense(64, activation=tf.nn.relu),
		layers.Dense(1)
	])
	model.compile(
		loss = 'mse',
		optimizer = keras.optimizers.RMSprop(0.001),		# 学习率初始值为0.001
		metrics = ['mae', 'mse']							# 评估指标: [平均绝对误差, 均方误差]
	)
	return model

''' 画训练中每个epoch后模型的评估指标值图
参数：history '''
def plot_history(history):
	hist = pd.DataFrame(history.history)	# 将history从dict类型转换为DataFrame类型
	hist['epoch'] = history.epoch			# 添加epoch列

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [MPG]')
	plt.plot(hist['epoch'], hist['loss'],		# 画平均绝对误差图
		label='Train Error')
	plt.legend()												# 添加图例

class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		rnd = epoch + 1
		if rnd % 5 == 0:
			print('.', end='')
		if rnd % 100 == 0:
			print('')

if __name__ == '__main__':
    main()