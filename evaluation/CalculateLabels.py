import os
import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.mixture import GaussianMixture as GMM

big_array = []
i = -10
while i <= 10 + 0.001:
    big_array.append(2**i)
    i += 3
print(big_array)

param_grid_svm = {
    'kernel': ['rbf', 'sigmoid'],
    'C': [pow(2, i) for i in range(-10, 7)],
    'gamma': [pow(2, i) for i in range(-10, 7)]
}

r_l1 = []
for i in range(-10, 0):
    j = pow(2, i)
    if 0.01 <= j <= 1:
        r_l1.append(j)

param_grid_els = {
    'alpha': [pow(2, i) for i in range(-10, 7)],
    'l1_ratio': r_l1
}

param_grid_gmm = {
    'n_components': [1, 2, 3, 4, 5, 6],
    'covariance_type': ['full', 'tied', 'diag', 'spherical']
}

param_name = {
    'SVM': ['C', 'gamma', 'kernel'],
    'ElasticNet': ['alpha', 'l1_ratio'],
    'GMM': ['covariance_type', 'n_components'],
}

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
    datasets = read_dataset(INPUTPATH)
    dic = {}
    for file in datasets:
        print(file)
        dic[file] = calculate_labels(alg, datasets[file])
        print()
    print(dic)
    df = pd.DataFrame(dic)
    df = df.transpose()

    ''' 保存标签 '''
    df.columns = param_name[alg]
    print(df.head())
    df.to_csv('knowledge/' + alg + '_labels.csv', index=False)
    print()

def get_param_name(alg):
    return param_name[alg]

def calculate_labels(alg, dataset):
    if alg == 'SVM':
        X = dataset.copy()
        y = X.pop('Label')

        model = SVC()
        gs_model = GridSearchCV(model, param_grid_svm, cv=5, iid=True)
        gs_model.fit(X, y)
        print(gs_model.best_params_)
        res = gs_model.best_params_
        print()
        return [res['C'], res['gamma'], res['kernel']]

    elif alg == 'ElasticNet':
        X = dataset.copy()
        y = X.pop('Label')

        model = ElasticNet()
        gs = GridSearchCV(model, param_grid_els, cv=5)
        gs.fit(X, y)
        print(gs.best_params_)
        res = gs.best_params_
        print()
        return [res['alpha'], res['l1_ratio']]

    elif alg == 'GMM':
        X = dataset.copy()
        y = X.pop('Label')
        
        model = GMM()
        gs = GridSearchCV(model, param_grid_gmm, cv=5, scoring='adjusted_rand_score')
        gs.fit(X, y)
        print(gs.best_params_)
        res = gs.best_params_
        print()
        return [[res['covariance_type']], res['n_components']]

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

