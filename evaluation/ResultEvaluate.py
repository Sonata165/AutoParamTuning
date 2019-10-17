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
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import adjusted_rand_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.mixture import GaussianMixture as GMM

from evaluation import CalculateLabels

warnings.filterwarnings('ignore')

def main():
    print('1. svm 2. elasticnet 3. gmm')
    choice = int(input('> '))
    if choice == 1:
        alg = 'SVM'
    elif choice == 2:
        alg = 'ElasticNet'
    elif choice == 3:
        alg = 'GMM'

    print(os.listdir('../system/input/'))
    datasets = read_dataset('../system/input/')

    predicted_df = pd.read_csv('../system/output/InitialResult.csv')
    x = predicted_df.pop('FileName')
    predicted_df.index = x
    predicted_df = predicted_df.transpose()
    # print(predicted_df)

    optimized_df = pd.read_csv('../system/output/FinalResult.csv')
    # print(optimized_df)

    ans = {}
    for file in datasets:
        print(file)
        t1 = time.time()
        best_param = CalculateLabels.calculate_labels(alg, datasets[file])
        print(best_param)
        t2 = time.time()
        print(predicted_df)
        predicted_param = list(predicted_df[file])
        optimized_param = list(optimized_df[file])
        print(predicted_param)
        print(optimized_param)

        best_score = score(alg, best_param, datasets[file])
        predicted_score = score(alg, predicted_param, datasets[file])
        optimized_score = score(alg, optimized_param, datasets[file])
        print(predicted_score, optimized_score)
        print()
        best_time = t2 - t1

        time_consuming = pd.read_csv('../system/output/time.csv').iloc[2]
        ours_time = time_consuming[file]
        ans[file] = [best_param, predicted_param,optimized_param, best_score, predicted_score, optimized_score,
            best_time, ours_time]
        print()
    ans_df = pd.DataFrame(ans)
    ans_df.index = ['best_param', 'net_param', 'opt_param', 'best_score', 'predicted_score', 'optimized_score',
            'best_time', 'ours_time']
    ans_df = ans_df.T
    ans_df.to_csv('Result_'+alg+'.csv')
    
    print()

def score(alg, params, dataset):
    '''
    以指定的参数在用户数据集上运行待调参算法，返回评判标准的得分
    Parameters:
      alg - 待调参算法
      params - 指定的参数列表
      dataset - 一个用户数据集，类型: DataFrame
    '''
    if alg == 'SVM':
        X = dataset.copy()
        y = X.pop('Label')
        t1 = float(params[0])
        t2 = float(params[1])
        t3 = params[2]
        if t1 < 0:
            t1 = 2**-10
        if t2 < 0:
            t2 = 2**-10
        model = SVC(C=t1, gamma=t2, kernel=t3)
        scores = cross_val_score(model, X, y, cv=5)
        ret = scores.mean()
    elif alg == 'ElasticNet':
        X = dataset.copy()
        y = X.pop('Label')
        model = ElasticNet(alpha=params[0], l1_ratio=params[1])
        scores = cross_val_score(model, X, y, cv=5)
        ret = scores.mean()
    elif alg == 'GMM':
        X = dataset.copy()
        y = X.pop('Label')
        y_pred=GMM(n_components=int(float(params[1])), covariance_type=params[0]).fit_predict(X)
        ret = adjusted_rand_score(y, y_pred)
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