'''
对system/input下数据集进行网格搜索寻找最优参数，
然后以最优参数运行待调参算法
再以某指定文件里的参数运行待调参算法
展示算法表现的比较，记录到evaluation/Result.csv
格式：


'''
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

sys.path.append('knowledge')

import CalculateLabels


def main():
    print('1. svm 2. elasticnet 3. gmm')
    choice = int(input('> '))
    if choice == 1:
        alg = 'SVM'
        INPUTPATH = '../database/SVM/'
    elif choice == 2:
        alg = 'ElasticNet'
        INPUTPATH = '../database/ElasticNet/'
    elif choice == 3:
        alg = 'GMM'
        INPUTPATH = '../database/GMM/'

    our_time = pd.read_csv('system/output/time.csv').iloc[2]

    datasets = read_dataset('system/input/')
    ans = {}
    for file in datasets:
        print(file)
        t1 = time.time()
        best_param = CalculateLabels.calculate_labels(alg, datasets[file])
        t2 = time.time()
        ours_param1, ours_param2 = get_our_param(file)
        best_score = score(alg, best_param, datasets[file])
        ours_score1 = score(alg, ours_param1, datasets[file])
        ours_score2 = score(alg, ours_param2, datasets[file])
        best_time = t2 - t1
        ours_time = our_time[file]
        ans[file] = [best_param, ours_param1, ours_param2, best_score, ours_score1, ours_score2,
            best_time, ours_time]
        print()
    ans_df = pd.DataFrame(ans)
    ans_df.index = ['best_param', 'ours_param1', 'ours_param2', 'best_score', 'ours_score1', 'ours_score2',
            'best_time', 'ours_time']
    ans_df.to_csv('evaluation/Result.csv')
    
    print()

def get_our_param(file):
    '''
    从系统的输出文件中获取对应数据集的参数
    Parameters:
      file - 数据集名称
    Returns:
      神经网络结果，一个参数的列表
      进一步优化结果，一个参数的列表
    '''
    param1 = pd.read_csv('system/output/InitialResult.csv')
    param2 = pd.read_csv('system/output/FinalResult.csv')
    return (list(param1[file]), list(param2[file]))

def score(alg, params, dataset):
    '''
    以指定的参数在用户数据集上运行待调参算法，返回评判标准的得分
    Parameters:
      alg - 待调参算法
      params - 指定的参数列表
      dataset - 一个用户数据集，类型: DataFrame
    '''
    if alg == 'SVM':
        print(dataset.head())
        X = dataset.copy()
        y = X.pop('Label')
        if params[0] < 0:
            params[0] = 2**-10
        if params[1] < 0:
            params[1] = 2**-10
        model = SVC(C=params[0], gamma=params[1])
        scores = cross_val_score(model, X, y, cv=5)
        ret = scores.mean()
    elif alg == 'ElasticNet':
        print()
    elif alg == 'GMM':
        print()
    return ret

def read_dataset(path):
    '''
    该函数读取path下所有数据集，
    Parameters:
      None - None
    Returns:
      一个字典，包含所有读入的数据集，格式如 数据集名称:数据集内容
      数据集类型为pandas.Dataframe
    '''
    print('读取数据集')
    files = os.listdir(path)
    datasets = {}
    for file in files:
        dataset = pd.read_csv(path + file, sep=',', skipinitialspace=True)
        datasets[file] = dataset
    return datasets

if __name__ == '__main__':
    main()