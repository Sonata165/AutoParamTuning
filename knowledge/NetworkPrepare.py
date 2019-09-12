'''
使用KnowledgePrepare生成的三个csv文件训练神经网络，
将训练好的模型保存至system/network下的三个.h5文件中，请用keras的model.save('路径名')
'''

from keras.layers import *
from keras import Model
import keras


def reg_net(input_shape):
    """
    生成单输出回归神经网
    :param input_shape: 输入维度，元组
    :return: compile好的Keras模型
    """
    x_input = Input(input_shape)
    x = Dense_withBN_Dropout(x_input, 16)
    x = Dense_withBN_Dropout(x, 16)
    x = Dense_withBN_Dropout(x, 4)
    x = Dense_withBN_Dropout(x, 1)
    model = Model(inputs=[x_input], outputs=[x])
    model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.Adam(0.001),  # 学习率初始值为0.001
        metrics=['mae', 'mse']  # 评估指标: [平均绝对误差, 均方误差]
    )
    return model


def classify_net(input_shape,output_dim):
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
    x = Dense_withBN_Dropout(x, output_dim, activation='softmax')
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


def build_GMM_ncomponents(input_shape,output_dim):
    """
    生成预测GMM超参数n_components的神经网
    :param input_shape: 输入维度，元组
    :param output_dim: 总类别数，int
    :return: compile好的Keras模型
    """
    return classify_net(input_shape, output_dim)


def build_GMM_covariance(input_shape, output_dim):
    """
    生成预测GMM超参数covariance的神经网
    :param input_shape: 输入维度，元组
    :param output_dim: 总类别数，int
    :return: compile好的Keras模型
    """
    return classify_net(input_shape, output_dim)


def Dense_withBN_Dropout(input, units, activation=None):
    """
    全连接-BN层-激活层-Dropout层的神经元模块
    :param input: 输入
    :param units: 全连接层神经元个数
    :param activation: 默认为None，则采用LeakyRelu激活函数，否则应传入Keras中激活函数的名字，如'softmax'
    :return: tensor，神经元输出
    """
    x = Dense(units=units)(input)
    x = BatchNormalization()(x)
    if activation is None:
        x = LeakyReLU(alpha=0.3)(x)
    else:
        x = Activation(activation)(x)
    x = Dropout(rate=0.1)(x)
    return x


class PrintDot(keras.callbacks.Callback):  # 一个回调函数
    def on_epoch_end(self, epoch, logs):
        rnd = epoch + 1
        if rnd % 5 == 0:
            print('.', end='')
        if rnd % 100 == 0:
            print('')


def train_nn(model, x_train, y_train, epochs, model_name):
    """
    训练神经网，保存神经网，返回history
    :param model: 要训练的模型
    :param x_train: x
    :param y_train: y   形式:[c,gamma]
    :param epochs: 起始训练轮数
    :param model_name: 拟合的算法名字
    :return: history
    """
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)  # 用EarlyStopping创建另一个回调函数
    history = model.fit(x=x_train, y=y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stop, PrintDot()])
    model.save('../system/network/' + model_name + '.h5')
    return history


def plot_history(history):
    # TODO
    pass

