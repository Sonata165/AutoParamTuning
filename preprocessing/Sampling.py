'''
对数据集进行抽样生成新的小数据集。
用[500-1000]的随机数来确定新的数据集大小
如果原数据集是新数据集大小的n * k倍，则抽样出新的n个数据集
注：在取样之前，还丢弃了有空缺的行
'''

import os
import pandas as pd
import shutil
import random

K = 2

def main():
    print('1. svm 2. elasticnet 3. gmm')
    choice = int(input('> '))
    if choice == 1:
        sampling('../database_init/SVM/', '../database_sample/SVM/')
    elif choice == 2:
        sampling('../database_init/ElasticNet/', '../database_sample/ElasticNet/')
    elif choice == 3:
        sampling('../database_init/GMM/', '../database_sample/GMM/')

def sampling(inputpath, outputpath):
    INPUTPATH = inputpath
    ERRORPATH = '../database_init/trouble_when_sampling/'
    OUTPUTPATH = outputpath
    files = os.listdir(INPUTPATH)
    print(files)
    for file in files:
        print(file)
        try:
            dataset = pd.read_csv(INPUTPATH + file, sep=',', skipinitialspace=True)
            dataset = dataset.dropna()
            n = len(dataset)
            N = random.randint(500, 1000)
            k = n // N

            if n > N:
                # for i in range(0, k):
                #     sp_dataset = dataset.sample(n=N)
                #     sp_dataset.to_csv(OUTPUTPATH +'sp_' + str(i) + '_' + file)
                if k <= 3:
                    sp_dataset = dataset.sample(n=N)
                    sp_dataset.to_csv(OUTPUTPATH +'sp_' + file, index=False)
                else:
                    for i in range(0, 3):
                        sp_dataset = dataset.sample(n=N)
                        sp_dataset.to_csv(OUTPUTPATH + 'sp_' + str(i) + '_' + file, index=False)
            else:
                dataset.to_csv(OUTPUTPATH +'sp_' + file, index=False)
            print('success')
        except UnicodeDecodeError:
            print('error')
            shutil.move(INPUTPATH + file, ERRORPATH + file)
        print()
        # sp_dataset = dataset.sample()

if __name__ == '__main__':
    main()