import os
import sys
import pandas as pd

sys.path.append('system')

from FeatureCalc import calculate_features

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
        dic[file] = calculate_features(alg, datasets[file])
        print()
    print(dic)
    df = pd.DataFrame(dic)
    df = df.transpose()
    print(df.head())
    df.to_csv('knowledge/' + alg + '_features.csv', index=False)
    print()

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

