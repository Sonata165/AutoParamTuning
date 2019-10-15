'''
检查各数据集的可行性
筛选../../database中的数据集，若表现过差，将该数据集移动到../../database/_OffSpec中
'''

import os
import pandas as pd
import shutil
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score

SVM_THRESHOLD = 0.6 # 当SVM分类正确率小于该值时，认为该数据集不合格
ELASTICNET_THRESHOLD = 0.1 # 当回归计算r2小于该值时，认为数据集不合格
GMM_THRESHOLD = 0.3 # 当GMM聚类

def main():
    '''
    主函数
    '''
    print('1. svm 2. elasticnet 3. gmm')
    choice = int(input('> '))
    if choice == 1:
        svm_filter()
    elif choice == 2:
        elastic_net_filter()
    elif choice == 3:
        gmm_filter()

def svm_filter():
    '''
    筛选SVM数据集
    '''
    DATASET_PATH = '../database/SVM/'
    OFFSPEC_PATH = '../database/_OffSpec/SVM/'
    datasets = read_dataset(DATASET_PATH)
    for filename in datasets:
        print(filename)
        try:
            result = svm_cross_validation(datasets[filename])
            print("得分:", result)
            if result < SVM_THRESHOLD:
                shutil.move(DATASET_PATH + filename, OFFSPEC_PATH + filename)
        except ValueError:
            print('error')
        
        print()

def elastic_net_filter():
    '''
    筛选ElasticNet数据集
    '''
    DATASET_PATH = '../database/ElasticNet/'
    OFFSPEC_PATH = '../database/_OffSpec/ElasticNet/'
    datasets = read_dataset(DATASET_PATH)
    for filename in datasets:
        print(filename)
        result = elastic_net_cross_validation(datasets[filename])
        print("得分:", result)
        if result < ELASTICNET_THRESHOLD:
            shutil.move(DATASET_PATH + filename, OFFSPEC_PATH + filename)
        print()

def gmm_filter():
    '''
    筛选GMM数据集
    '''
    DATASET_PATH = '../database/GMM/'
    OFFSPEC_PATH = '../database/_OffSpec/GMM/'
    datasets = read_dataset(DATASET_PATH)
    for filename in datasets:
        print(filename)
        result = gmm_score(datasets[filename])
        print("得分:", result)
        if result < GMM_THRESHOLD:
            shutil.move(DATASET_PATH + filename, OFFSPEC_PATH + filename)
        print()

def svm_cross_validation(dataset):
    '''
    SVM交叉验证
    Parameters:
      待跑SVM的数据集，格式：pandas.DataFrame
    Returns:
      数据集10折交叉验证得分，使用默认的SVM超参数，评估标准：正确率
    '''
    y = dataset.pop('Label')
    X = dataset

    model = SVC(gamma='auto')
    scores = cross_val_score(model, X, y, cv=10)

    ret = scores.mean()
    return ret

def elastic_net_cross_validation(dataset):
    '''
    ElasticNet交叉验证
    Parameters:
      待运行ElasticNet的数据集，格式：pandas.DataFrame
    Returns:
      数据集10折交叉验证得分，使用默认的ElasticNet超参数，评估标准：r2
    '''
    print()

    y = dataset.pop('Label')
    X = dataset

    model = ElasticNet()
    model.fit(X, y)
    scores = cross_val_score(model, X, y, cv=10, scoring='r2')

    ret = scores.mean()
    return ret

def gmm_score(dataset):
    '''
    GMM聚类效果评估
    Parameters:
      待运行GMM的数据集，格式：pandas.DataFrame
    Returns:
      数据集聚类结果，使用已知的簇的个数，其他超参数使用默认值，评估标准：adjusted_rand_score
    '''
    y_true = dataset.pop('Label')
    X = dataset

    y_pred = GMM(n_components=3, covariance_type='full').fit_predict(X)

    ret = adjusted_rand_score(y_true, y_pred)
    return ret

def read_dataset(path):
    '''
    该函数读取path下所有数据集，
    Parameters:
      path - 要读入的数据集所在目录
    Returns:
      一个字典，包含所有读入的数据集，格式如 数据集名称:数据集内容
        数据集类型为pandas.Dataframe
    '''
    files = os.listdir(path)
    datasets = {}
    print(files)
    for file in files:
        print(file)
        dataset = pd.read_csv(path + file, sep=',', skipinitialspace=True)
        datasets[file] = dataset
    return datasets

def drop_na_within_folder(path):
    '''
    读取path文件夹下所有数据集，剔除空行，然后保存至原路径.
    Parameters:
      path - 将要剔除空行的数据集们所在的目录
    '''
    files = os.listdir(path)
    for file in files:
        dataset = pd.read_csv(path + file)
        dataset = dataset.dropna()
        dataset.to_csv(path + file)

def drop_all_na():
    '''
    剔除'../database'下所有数据集（不包括_OffSpec目录）的空行
    '''
    drop_na_within_folder('../database/SVM/')
    drop_na_within_folder('../database/ElasticNet/')
    drop_na_within_folder('../database/GMM/')

if __name__ == '__main__':
    main()