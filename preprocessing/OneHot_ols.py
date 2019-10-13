import os
import pandas as pd

def main():
    print('1. svm 2. elasticnet 3. gmm')
    choice = int(input('> '))
    if choice == 1:
        one_hot(choice, '../database_sample/SVM/', '../database/SVM/')
    elif choice == 2:
        one_hot(choice, '../database_sample/ElasticNet/', '../database/ElasticNet/')
    elif choice == 3:
        one_hot(choice, '../database_sample/GMM/', '../database/GMM/')

def one_hot(choice, inputpath, outputpath):
    INPUTPATH = inputpath
    OUTPUTPATH = outputpath
    files = os.listdir(INPUTPATH)
    datasets = {}
    for file in files:
        print(file)
        dataset = pd.read_csv(INPUTPATH + file, sep=',', skipinitialspace=True)
        dataset = dataset.dropna()
        # print(dataset.head())
        y = dataset.pop('Label')
        dataset_onehot = pd.get_dummies(dataset)
        print(len(dataset_onehot))
        if (choice == 1):
            dataset_onehot['Label'] = pd.factorize(y)[0]
        dataset_onehot.to_csv(OUTPUTPATH + file, index=False)

if __name__ == '__main__':
    main()