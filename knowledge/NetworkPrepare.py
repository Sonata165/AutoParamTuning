'''
使用KnowledgePrepare生成的三个csv文件训练神经网络，
将训练好的模型保存至system/network下的三个.h5文件中，请用keras的model.save('路径名')
'''

import multiprocessing as mp
from keras.layers import *
from keras import Model
import keras
import matplotlib.pyplot as plt
import pandas as pd
from math import *
from sklearn.preprocessing import StandardScaler


def read_data(modelName, path="knowledge/"):
    """
    从/knowledge/modelName.csv读取数据
    :param modelName: 模型名称，读取文件的名称
    :param path: 要读取的文件所在的路径
    :return: train_X, train_y, label顺序按照KnowledgePrepare中get_param_name的顺序，x做标准化处理
    """
    raw = pd.read_csv(path + modelName + ".csv")
    if modelName == "SVM":
        y = raw.values[:, -6:]
        x = StandardScaler().fit_transform(raw.values[:, :-6])
    elif modelName == "ElasticNet":
        y = raw.values[:, -2:]
        x = StandardScaler().fit_transform(raw.values[:, :-2])
    elif modelName == "GMM":
        y = raw.values[:, -5:]
        x = StandardScaler().fit_transform(raw.values[:, :-5])
    else:
        return None, None
    # y = np.array(y)

    """
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    return train_x, test_x, train_y, test_y
    """
    return x, y


def reg_net(input_shape, activation=None):
    """
    生成单输出回归神经网
    :param input_shape: 输入维度，元组
    :param activation: 激活函数，默认为None，如要指定激活函数，传入length为4的元组，元素为keras.layer中的激活函数层，如keras.layers.ReLU()
    :return: compile好的Keras模型
    """
    x_input = Input(input_shape)
    x = Dense_withBN_Dropout(x_input, 16, activation)
    x = Dense_withBN_Dropout(x, 16, activation)
    x = Dense_withBN_Dropout(x, 4, activation)
    x = Dense_withBN_Dropout(x, 1, activation)
    model = Model(inputs=[x_input], outputs=[x])
    model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.Adam(0.001),  # 学习率初始值为0.001
        metrics=['mae', 'mse']  # 评估指标: [平均绝对误差, 均方误差]
    )
    return model


def classify_net(input_shape, output_dim):
    """
    创建一个分类Keras模型
    :param input_shape: 输入维度，元组，(特征数,)
    :param output_dim: 总类别数
    :return: compile好的模型
    """
    x_input = Input(input_shape)
    x = Dense_withBN_Dropout(x_input, 16)
    x = Dense_withBN_Dropout(x, 16)
    x = Dense_withBN_Dropout(x, 4)
    x = Dense_withBN_Dropout(x, output_dim, activation=Softmax())
    model = Model(inputs=[x_input], outputs=[x])
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(0.001),  # 学习率初始值为0.001
        metrics=['mae', 'mse']  # 评估指标: [平均绝对误差, 均方误差]
    )
    return model


def build_SVM_Kernel_nn(input_shape, output_dim):
    """
    生成预测SVM超参数Kernel的神经网
    :param input_shape: 输入维度，元组，如有6个feature，则input_shape=(6,)
    :param output_dim: 输出维度，int，分类问题中的总类别数，如有4种核函数，则output=4
                        输出如下：[0,1,0,0]，值为1的为预测的kernel，列表按kernel名称的字典序排列
    :return: compile好的keras模型
    """
    return classify_net(input_shape, output_dim)


def build_SVM_C_nn(input_shape):
    """
    生成预测SVM超参数C的神经网
    :param input_shape: 输入维度，元组，如有6个feature，则input_shape=(6,)
    :return: compile好的Keras模型
    """
    return reg_net(input_shape)


def build_SVM_gamma_nn(input_shape):
    """
    生成预测SVM超参数gamma的神经网
    :param input_shape: 输入维度，元组，如有6个feature，则input_shape=(6,)
    :return: compile好的Keras模型
    """
    return reg_net(input_shape)


def build_ElasticNet_alpha_nn(input_shape):
    """
    生成预测ElasticNet超参数alpha的神经网
    :param input_shape: 输入维度，元组，如有6个feature，则input_shape=(6,)
    :return: compile好的Keras模型
    """
    return reg_net(input_shape)


def build_ElasticNet_l1ratio_nn(input_shape):
    """
    生成预测ElasticNet超参数alpha的神经网
    :param input_shape: 输入维度，元组，如有6个feature，则input_shape=(6,)
    :return: compile好的Keras模型
    """
    return reg_net(input_shape)


def build_GMM_n_components(input_shape, output_dim):
    """
    生成预测GMM超参数n_components的神经网
    :param input_shape: 输入维度，元组
    :param output_dim: 总类别数，int
    :return: compile好的Keras模型
    """
    return classify_net(input_shape, output_dim)


def build_GMM_covariance_type(input_shape):
    """
    生成预测GMM超参数covariance_type的神经网
    注意，这里神经网的输出结果要转换为整数使用
    :param input_shape: 输入维度，元组
    :return: compile好的Keras模型
    """
    return reg_net(input_shape)


def Dense_withBN_Dropout(input, units, activation=None):
    """
    全连接-BN层-激活层-Dropout层的神经元模块
    :param input: 输入
    :param units: 全连接层神经元个数
    :param activation: 默认为None，采用LeakyRelu激活函数，否则应传入Keras中的激活函数，如keras.layers.ReLU()
    :return: tensor，神经元输出
    """
    x = Dense(units=units)(input)
    x = BatchNormalization()(x)
    if activation is None:
        x = LeakyReLU(alpha=0.3)(x)
    else:
        x = activation(x)
    # x = Dropout(rate=0.1)(x)
    return x


class PrintDot(keras.callbacks.Callback):  # 一个回调函数
    def on_epoch_end(self, epoch, logs):
        rnd = epoch + 1
        if rnd % 5 == 0:
            print('.', end='')
        if rnd % 100 == 0:
            print('')


def train_nn(model, x_train, y_train, epochs, model_name, save_path='../system/network/'):
    """
    训练神经网，保存神经网
    使用多进程运行，训练结束后将训练好的模型保存到h5文件，
    并画出训练历史
    :param model: 要训练的模型
    :param x_train: x
    :param y_train: y
    :param epochs: 起始训练轮数
    :param model_name: string, 要保存的名字，建议命名为:算法名_超参数名
    :param save_path: 保存模型的路径，例如：'../system/network/'
    :return: None
    """
    if model_name == "SVM_C":
        y_train = y_train[:, 0]
    elif model_name == "SVM_gamma":
        y_train = y_train[:, 1]
    elif model_name == "SVM_kernel":
        y_train = y_train[:, 2:]
    elif model_name == "ElasticNet_alpha":
        y_train = y_train[:, 0]
    elif model_name == "ElasticNet_l1_ratio":
        y_train = y_train[:, 1]
    elif model_name == "GMM_n_components":
        y_train = y_train[:, :-1]
    elif model_name == "GMM_covariance_type":
        y_train = y_train[:, -1]
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)  # 用EarlyStopping创建另一个回调函数
    history = model.fit(x=x_train, y=y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stop, PrintDot()])
    model.save(save_path + model_name + '.h5')
    # plot_history(history, model_name)


def build_nn_for_model(modelName, input_shape=None, output_dim=None):
    """
    对要训练的算法建立一系列神经网分别用来预测各个超参数
    :param modelName: 算法名称
    :param input_shape: shape()
    :param output_dim: 分类的总类别数int
    :return: {算法名_超参数名：对应的nn}
    """
    nn = {}
    if modelName == "SVM":
        nn['SVM_C'] = build_SVM_C_nn(input_shape)
        nn['SVM_gamma'] = build_SVM_gamma_nn(input_shape)
        nn['SVM_kernel'] = build_SVM_Kernel_nn(input_shape, output_dim)
    elif modelName == "ElasticNet":
        nn['ElasticNet_alpha'] = build_ElasticNet_alpha_nn(input_shape)
        nn['ElasticNet_l1_ratio'] = build_ElasticNet_l1ratio_nn(input_shape)
    elif modelName == "GMM":
        nn['GMM_n_components'] = build_GMM_n_components(input_shape, output_dim)
        nn['GMM_covariance_type'] = build_GMM_covariance_type(input_shape)
    return nn


def spatial_pyramid_pooling(input, output_dim):
    """
    空间金字塔池化，将不同shape的input池化为同尺寸的输出，池化按照四等分，两等分，一等分，这里先使用平均池化方法
    :param input: 输入数据，应为整个数据集的原始数据，格式为二维numpy，shape[0]为不同数据，shape[1]为原始数据特征
    使用pandas中的DataFrame.values即可获得
    :param output_dim: 输出维度，int，表示将整个数据集化为(output_dim, output_dim)大小的特征图
    :return: 池化后的结果
    """
    div = [4., 2., 1.]
    for base in div:
        # 计算窗口大小和步长
        window_len = ceil(input.shape[0] / base)
        window_wid = ceil(input.shape[1] / base)
        stripe_len = floor(input.shape[0] / base)
        stripe_wid = floor(input.shape[1] / base)
        # 截取窗口数据
        # 起始点设置在[0,0]
        start_wid = 0
        start_len = 0
        data = input[start_wid:start_wid + window_wid+1, start_len:start_len + window_len+1]
        # 池化
        data = data.mean
        # 计算下一个窗口位置
        start_wid += stripe_wid
        start_len += stripe_len


def build_encoder(input, output_dim):
    pass


def train_test_nn_for_model(modelName, epoch, train_X, train_y, input_shape=None, output_dim=None,
                            save_path='../system/network/'):
    """
    针对算法创建神经网
    使用多进程并行训练针对一个算法不同超参数的神经网
    保存训练结果，获取训练历史，并进行test
    :param modelName: 要训练的算法名字
    :param epoch: 训练轮数
    :param train_X: 训练用特征
    :param train_y: 训练用标签
    :param input_shape: 神经网输入的shape，类型与numpy中shape相同，默认为None
    :param output_dim: 输出维度，只有包含分类神经网时有效，值等于总类别数目，默认为None
    :param save_path: 保存模型的路径，例如：'../system/network/'
    :return:
    """
    print("开始创建神经网")
    nn_dict = build_nn_for_model(modelName, input_shape, output_dim)
    jobs = []
    for name, nn in nn_dict.items():
        job = mp.Process(target=train_nn, args=(nn, train_X, train_y, epoch, name, save_path))
        jobs.append(job)
        job.start()
    for job in jobs:
        job.join()


def plot_history(history, param_name):
    hist = pd.DataFrame(history.history)  # 将history从dict类型转换为DataFrame类型
    print(hist)
    hist['epoch'] = history.epoch  # 添加epoch列

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [' + param_name + ']')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],  # 画平均绝对误差图
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],  # 画验证集平均绝对误差图
             label='Val Error')
    plt.legend()  # 添加图例
    # plt.ylim([0,100])												# 设置y轴范围

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$' + param_name + '^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],  # 画均方误差图
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],  # 画验证集均方误差图
             label='Val Error')
    plt.legend()  # 添加图例
# plt.ylim([0,32000])											# 设置y轴范围
