'''
本模块对数据集进行预处理
四个步骤：
1. 去掉空缺行
2. 抽样
3. one-hot编码
'''

import sys

sys.path.append('preprocessing')

import OneHot
import 

def main():
    

def sampling():
    INPUTPATH = '../database_init'
    ERRORPATH = 'preprocessing/problem/'
    OUTPUTPATH = 'preprocessing/sp/'
    files = os.listdir(INPUTPATH)
    print(files)
    for file in files:
        print(file)
        try:
            dataset = pd.read_csv(INPUTPATH + file, sep=',', skipinitialspace=True)
            dataset = dataset.dropna()
            n = len(dataset)
            N = random.randint(500, 1000)
            k = n // N // 2 # 抽样遍数

            if n > N:
                # for i in range(0, k):
                #     sp_dataset = dataset.sample(n=N)
                #     sp_dataset.to_csv(OUTPUTPATH +'sp_' + str(i) + '_' + file)
                sp_dataset = dataset.sample(n=N)
                sp_dataset.to_csv(OUTPUTPATH +'sp_' + file)
            else:
                dataset.to_csv(OUTPUTPATH +'sp_' + file)
            print('success')
        except UnicodeDecodeError:
            print('error')
            shutil.move(INPUTPATH + file, ERRORPATH + file)
        print()