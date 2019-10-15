import os
# rootPath = os.path.realpath(__file__).replace('\AutoParamTuning\preprocessing\OneHot.py','')
# initPath = rootPath + os.sep + 'database_init'
# destinyPath = rootPath + os.sep + 'database'
import numpy as np
import pandas as pd
import math

initPath = None
destinyPath = None

def main():
    print('1. svm 2. elasticnet 3. gmm')
    choice = int(input('> '))
    if choice == 1:
        one_hot('../database_sample/SVM', '../database/SVM')
    elif choice == 2:
        one_hot('../database_sample/ElasticNet', '../database/ElasticNet')
    elif choice == 3:
        one_hot('../database_sample/GMM', '../database/GMM')

def one_hot(input_path, output_path):
    '''
    将input_path下的数据集进行onehot编码，保存到output_path目录下
    注意：input_path, output_path末尾没有反斜杠
    '''
    print('inputpath:', input_path)
    work(input_path, output_path)

'''
work方法
整体操作
'''
def work(initPath, destinyPath):
    import os
    files = os.listdir(initPath)
    print(files)
    print('shit')
    for filename in files:
        if filename.__contains__('.csv'):
            print(filename)
            # try: 
            #     singleFileProcessAndWrite(singleFileReadAndProcess(filename),filename)
            # except ValueError:
            #     print('error')
            singleFileProcessAndWrite(destinyPath, singleFileReadAndProcess(initPath, filename),filename)
            print()
    return
'''
处理单个文件
将数据从文件中读取并进行处理
'''
def singleFileReadAndProcess(initPath, filename):
    realPath = initPath + os.sep + filename
    import pandas as pd
    data = pd.read_csv(realPath,encoding = 'gbk').values
    columnNum = len(data[0])
    rowNum = len(data)
    '''
    ans
    将结果保存在ans中
    '''
    ans = np.zeros(shape = (rowNum,0))
    '''
    dic记录每一个字符串的二进制编码
    '''
    for j in range(columnNum):
        if judge(data,j,rowNum):
            tmp = np.zeros(shape = rowNum)
            for i in range(rowNum):
                tmp[i] = data[i][j]
            ans = np.insert(ans,len(ans[0]),values = tmp,axis = 1)
        else:
            if j == columnNum - 1:
                dic = {}
                cnt = 0
                for i in range(rowNum):
                    if dic.__contains__(data[i][j]):
                        continue
                    dic[data[i][j]] = cnt
                    cnt = cnt + 1
                tmp = np.zeros(shape = rowNum)
                for i in range(rowNum):
                    tmp[i] = dic[data[i][j]]
                ans = np.insert(ans,len(ans[0]),values = tmp,axis = 1)
                continue
            dic = {}
            cnt = 0
            for i in range(rowNum):
                if dic.__contains__(data[i][j]):
                    continue
                dic[data[i][j]] = cnt
                cnt = cnt + 1
            if cnt == 1:
                length = 1
            else:
                lenth = int(math.log2(cnt - 1)) + 1
            tmp = np.zeros(shape = (lenth,rowNum))
            for i in range(rowNum):
                things = decimalToBinaryArray(dic[data[i][j]],lenth)
                for k in range(lenth):
                    tmp[k][i] = things[k]
            ans = np.insert(ans,len(ans[0]),values = tmp,axis = 1)
    return ans
'''
处理单个文件
将数据处理并写进文件
'''
def singleFileProcessAndWrite(destinyPath, data,filename):
    realPath = destinyPath + os.sep + filename
    finalData = {}
    rowNum = len(data)
    for j in range(len(data[0])):
        tmp = []
        for i in range(rowNum):
            tmp.append(data[i][j])
        if j == len(data[0]) - 1:
            finalData['Label'] = tmp
            continue
        finalData[j] = tmp
    df_content = pd.DataFrame(finalData)
    df_content.to_csv(realPath,index = False)
    return
'''
将一个十进制数字转换成二进制并保存在数组中
'''
def decimalToBinaryArray(x,lenth):
    import numpy as np
    ret = np.zeros(shape = (lenth))
    index = lenth - 1
    while x != 0:
        ret[index] = x % 2
        x = int(x / 2)
        index = index - 1
    return ret
'''
判断是不是这一列都是数字
'''
def judge(data,j,rowNum):
    for i in range(rowNum):
        if isnub(data[i][j]) == False:
            return False
    return True
'''
判断是不是数字
'''
def isnub(s):
    try:
        nb = float(s) #将字符串转换成数字成功则返回True
        return True
    except ValueError as e:
        return False


if __name__ == '__main__':
    main()

