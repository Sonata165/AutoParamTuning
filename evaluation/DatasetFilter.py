'''
检查各数据集的可行性
筛选../../database中的数据集，若表现过差，将该数据集移动到../../database/_OffSpec中
'''

import os
import pandas as pd
import shutil
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

SVM_THRESHOLD = 0.7 # 当SVM分类正确率小于该值时，认为该数据集不合格

def main():
    '''
    主函数
    '''
    svm_filter()
    elastic_net_filter()
    gmm_filter()

def svm_filter():
    '''
    筛选SVM数据集
    '''
    datasets = read_dataset('../database/SVM/')
    for filename in datasets:
        print(filename)
        result = svm_cross_validation(datasets[filename])
        print("得分:", result)
        if result < SVM_THRESHOLD:
            shutil.move('../database/SVM/' + filename, '../database/_OffSpec/SVM/' + filename)
        print()

def elastic_net_filter():
    '''
    筛选ElasticNet数据集
    '''
    datasets = read_dataset('../database/ElasticNet/')
    for filename in datasets:
        print(filename)
        print
        print()

def gmm_filter():
    '''
    筛选GMM数据集
    '''
    print('TODO')

def svm_cross_validation(dataset):
    '''
    SVM交叉验证
    Parameters:
      待跑SVM的数据集，格式：pandas.DataFrame
    Returns:
      数据集10折交叉验证得分
    '''
    y = dataset.pop('Label')
    X = dataset

    model = SVC(gamma='auto')
    scores = cross_val_score(model, X, y, cv=10)

    ret = scores.mean()
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
    print('读取数据集')
    files = os.listdir(path)
    datasets = {}
    for file in files:
        dataset = pd.read_csv(path + file, sep=',', skipinitialspace=True)
        datasets[file] = dataset
    return datasets

if __name__ == '__main__':
    main()