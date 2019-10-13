'''
Author: 欧龙燊
客户端
前提：用户的数据集已经转化为数值类型，已经标准化
注意：我们的数据集还未被标准化
本模块读入input目录下所有数据集，进行参数调优后将结果保存在output中
要求用户同时放入input文件夹下的所有数据集适用于同一种机器学习算法
保存格式：csv文件，第一列为数据集名称，第一行为最优参数名称
'''

import os
import time
import numpy as np
import pandas as pd
import sys
from tensorflow import keras

sys.path.append('knowledge')

import FeatureCalc
import FurtherOpt
import KnowledgePrepare
import CalculateLabels

def main():
    '''
    主函数
    '''
    print("选择算法")
    print("1. SVM分类")
    print("2. 弹性网络回归")
    print("3. GMM聚类")
    choice = int(input("> "))
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
    for filename in datasets: # filename是各数据集名
        features = FeatureCalc.calculate_features(alg_name, datasets[filename])
        feature_df[filename] = features
    # print('用户数据集特征：')
    # print(feature_df)
    feature_df.to_csv('system/output/Features.csv', index=False)

    ''' 将特征送入神经网络计算得到预测结果 '''
    # TODO 这里按照神经网络只有一个输出来写
    predicted_df = pd.DataFrame()
    param_names = KnowledgePrepare.get_param_name(alg_name)
    predicted_df['ParamName'] = param_names

    ans = {}
    params = CalculateLabels.get_param_name(alg_name)

    time_consume = {} # 保存用时，第一行是神经网络时间，第二行是进一步优化时间，第三行是总时间
    
    for filename in datasets:
        print(filename)
        ans[filename] = []
        time_consume[filename] = []
        t1 = time.time() # 放入神经网络前的时间点
        for param in params:
            print(param)
            
            model = keras.models.load_model('system/network/' + alg_name + '_' + param + '.h5')
            result = model.predict(np.expand_dims(feature_df[filename], 0))[0]
            
            if len(result) > 1:
                result = np.argmax(result)
            else:
                result = result[0]
            print(result)
            ans[filename].append(result)
        t2 = time.time() # 放入神经网络后的时间点
        time_consume[filename].append(t2 - t1)
        # model_gamma = keras.models.load_model('system/network/SVM_gamma.h5')
        # result = model_gamma.predict(np.expand_dims(feature_df[filename], 0))
        # ans[filename].append(result[0][0])
    print('神经网络预测结果为：')
    predicted_df = pd.DataFrame(ans)
    # predicted_df.index = params
    print(predicted_df)

    if alg_name == 'GMM':
        conv_type = predicted_df.iloc[0]
        ans = []
        print(conv_type)
        for i in range(0, len(conv_type)):
            print(i)
            print(type(i))
            ans.append(CalculateLabels.iTos_gmm_covariance_type[conv_type[i]])
        predicted_df.iloc[0] = ans
        print(predicted_df)

    predicted_df.to_csv('system/output/InitialResult.csv', index=False)


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
    time_df.to_csv('system/output/time.csv', index=False)

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
    INPUTPATH = 'system/input/'
    files = os.listdir(INPUTPATH)
    datasets = {}
    for file in files:
        dataset = pd.read_csv(INPUTPATH + file, sep=',', skipinitialspace=True)
        datasets[file] = dataset
    return datasets


if __name__ == '__main__':
    main()
