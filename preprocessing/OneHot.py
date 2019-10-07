#'''
#对数据集进行one_hot编码
#注意！OneHot编码前的数据集所在目录命名为database_init
#本模块处理database_old下除了_OffSpec下之外的数据集，并以相同结构保存在database目录中
#项目结构：
#- AutoParamTuning
#- database_init
#- database
#以上三个文件夹放在在同一目录下
#'''
#
#def work():
#    '''
#    完成one-hot编码
#    '''
import os
rootPath = os.path.realpath(__file__).replace('\AutoParamTuning\preprocessing\OneHot.py','')
initPath = rootPath + os.sep + 'database_init'
destinyPath = rootPath + os.sep + 'database'
import numpy as np
import pandas as pd
import math
'''
work方法
整体操作
'''
def work():
    import os
    files = os.listdir(initPath)
    for filename in files:
        if filename.__contains__('.csv'):
            singleFileProcessAndWrite(singleFileReadAndProcess(filename),filename)
    return
'''
处理单个文件
将数据从文件中读取并进行处理
'''
def singleFileReadAndProcess(filename):
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
def singleFileProcessAndWrite(data,filename):
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



