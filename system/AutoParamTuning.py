'''
Author: 欧龙燊
客户端
前提：用户的数据集已经转化为数值类型，已经标准化
注意：我们的数据集还未被标准化
本模块读入input目录下所有数据集，进行参数调优后将结果保存在output中
要求用户同时放入input文件夹下的所有数据集适用于同一种机器学习算法
保存格式：csv文件，第一列为数据集名称，第一行为最优参数名称
'''
import json
import os
import time
import numpy as np
import pandas as pd
import sys
import keras
from math import *
# from tensorflow import keras
# sys.path.append('knowledge')

import system.FeatureCalc
import system.FurtherOpt
import knowledge.KnowledgePrepare
import knowledge.CalculateLabels
from knowledge import KnowledgePrepare, CalculateLabels, NetworkPrepare
from system import FeatureCalc


def main():
    '''
    主函数
    '''
    print("选择算法")
    print("1. SVM分类")
    print("2. 弹性网络回归")
    print("3. GMM聚类")
    choice = int(input("> "))
    alg_name = None
    if choice == 1:
        alg_name = 'SVM'
    elif choice == 2:
        alg_name = 'ElasticNet'
    elif choice == 3:
        alg_name = 'GMM'

    datasets = read_dataset()

    ''' 计算数据集特征并保存到Features.csv '''
    feature_df = pd.DataFrame()
    feature_names = FeatureCalc.get_feature_name(alg_name)
    feature_df['FeatureName'] = feature_names
    for filename in datasets:  # filename是各数据集名
        features = FeatureCalc.calculate_features(alg_name, datasets[filename])
        feature_df[filename] = features
    # print('用户数据集特征：')
    # print(feature_df)
    feature_df.to_csv('output/Features.csv', index=False)
    '''对特征进行反标准化'''
    std_x = None
    if alg_name == "SVM":
        f = open("../knowledge/" + 'SVM_std_x.json', 'r').read()
        std_x = json.loads(f)
    elif alg_name == "ElasticNet":
        f = open("../knowledge/" + 'Elastic_std_x.json', 'r').read()
        std_x = json.loads(f)
    elif alg_name == "GMM":
        f = open("../knowledge/" + 'GMM_std_x.json', 'r').read()
        std_x = json.loads(f)
    data = feature_df.values
    for i in range(0,data.shape[0]):
        for j in range(1,data.shape[1]):
            data[i][j] = data[i][j] * sqrt(std_x['var'][j]) + std_x['mean'][j]
    feature_df = pd.DataFrame(data, index=feature_df.index, columns=feature_df.columns)
    ''' 将特征送入神经网络计算得到预测结果 '''

    predicted_df = pd.DataFrame()
    param_names = KnowledgePrepare.get_param_name(alg_name)
    predicted_df['ParamName'] = param_names

    ans = pd.DataFrame()
    params = CalculateLabels.get_param_name(alg_name)

    time_consume = {}  # 保存用时，第一行是神经网络时间，第二行是进一步优化时间，第三行是总时间

    for filename in datasets:
        print(filename)
        time_consume[filename] = []
        t1 = time.time()  # 放入神经网络前的时间点
        for param in params:
            print(param)
            # 读取神经网
            result = {}
            if alg_name == "SVM":
                result['SVM_C'] = \
                    keras.models.load_model('network/SVM_C.h5').predict(np.expand_dims(feature_df[filename], 0))[0][0]
                result['SVM_gamma'] = \
                    keras.models.load_model('network/SVM_gamma.h5').predict(np.expand_dims(feature_df[filename], 0))[0][
                        0]
                index = np.argmax(keras.models.load_model('network/SVM_kernel.h5').predict(
                    np.expand_dims(feature_df[filename], 0))[0])
                kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']
                result['SVM_kernel'] = kernel_list[index]
            elif alg_name == "ElasticNet":
                result['ElasticNet_alpha'] = keras.models.load_model('network/ElasticNet_alpha.h5').predict(
                    np.expand_dims(feature_df[filename], 0))[0][0]
                result['ElasticNet_l1_ratio'] = keras.models.load_model('network/ElasticNet_l1_ratio.h5').predict(
                    np.expand_dims(feature_df[filename], 0))[0][0]
            elif alg_name == "GMM":
                components = int(keras.models.load_model('network/GMM_n_components.h5').predict(
                    np.expand_dims(feature_df[filename], 0))[0][0])
                result['GMM_n_components'] = 1 if components == 0 else components
                index = np.argmax(keras.models.load_model('network/GMM_covariance_type.h5').predict(
                    np.expand_dims(feature_df[filename], 0))[0])
                covariance_list = ['full', 'tied', 'diag', 'spherical']
                result['GMM_covariance_type'] = covariance_list[index]
            print(result)
            ans = ans.append(pd.Series(result, name=filename))
        t2 = time.time()  # 放入神经网络后的时间点
        time_consume[filename].append(t2 - t1)
        # model_gamma = keras.models.load_model('system/network/SVM_gamma.h5')
        # result = model_gamma.predict(np.expand_dims(feature_df[filename], 0))
        # ans[filename].append(result[0][0])
    print('神经网络预测结果为：')
    predicted_df = ans
    # predicted_df.index = params
    print(predicted_df)
    predicted_df.to_csv('output/InitialResult.csv')

    # ''' 进一步优化预测结果，得到最终结果 '''
    # optimized_df = pd.DataFrame()
    # optimized_df['ParamName'] = param_names
    # for filename in datasets:
    #     predicted_param = ans[filename]
    #     optimized_param = FurtherOpt.find_local_supreme(datasets[filename], alg_name, predicted_param)
    #     optimized_df[filename] = optimized_param
    # print('进一步优化结果为：')
    # print(optimized_df)
    # optimized_df.to_csv('system/output/FinalResult.csv', index=False)

    # ''' 计算总时间 '''
    # for file in time:
    #     time[file].append(time[file][0] + time[file][1])

    time_df = pd.DataFrame(time_consume)
    time_df.to_csv('output/time.csv', index=False)

    print('程序结束！')


def read_dataset():
    '''
    该函数读取system/input下所有数据集，
    Parameters:
      None - None
    Returns:
      一个字典，包含所有读入的数据集，格式如 数据集名称:数据集内容
      数据集类型为pandas.Dataframe
    '''
    print('读取数据集')
    INPUTPATH = 'input/'
    files = os.listdir(INPUTPATH)
    datasets = {}
    for file in files:
        dataset = pd.read_csv(INPUTPATH + file, sep=',', skipinitialspace=True)
        datasets[file] = dataset
    return datasets


def read_data(modelName, path="knowledge/"):
    """
    从/knowledge/modelName.csv读取数据，并将标准化所使用的对象序列化保存到path目录下
    :param modelName: 模型名称，读取文件的名称
    :param path: 要读取的文件所在的路径
    :return: train_X, train_y, label顺序按照KnowledgePrepare中get_param_name的顺序，x,y做标准化处理
    """
    raw = pd.read_csv(path + modelName + ".csv")
    if modelName == "SVM":
        with open("knowledge/" + 'SVM_std_y.json', 'w') as f:
            std_y = json.load(f)
        with open("knowledge/" + 'SVM_std_x.json', 'w') as f:
            std_x = json.load(f)
        y = std_y.fit_transform(raw.values[:, -6:-4])
        y = np.hstack((y, raw.values[:, -4:]))
        x = std_x.fit_transform(raw.values[:, :-6])

    elif modelName == "ElasticNet":
        with open("knowledge/" + 'Elastic_std_y.json', 'w') as f:
            std_y = json.load(f)
        with open("knowledge/" + 'Elastic_std_x.json', 'w') as f:
            std_x = json.load(f)
        y = std_y.fit_transform(raw.values[:, -2:])
        x = std_x.fit_transform(raw.values[:, :-2])
    elif modelName == "GMM":
        with open("knowledge/" + 'GMM_std_x.json', 'w') as f:
            std_x = json.load(f)
        y = raw.values[:, -5:]
        x = std_x.fit_transform(raw.values[:, :-5])
    else:
        return None, None
    return x, y


if __name__ == '__main__':
    main()
