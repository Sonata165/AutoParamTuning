# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:22:46 2019

@author: 陈泊舟
"""
import pandas as pd
import math
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score

import os

params_path = os.path.realpath(__file__).replace('\system\FutherOpt.py', '') + os.sep + 'knowledge'
import sys

sys.path.append(params_path)

import knowledge.KnowledgePrepare

'''
均在fin_local_supreme函数刚开始时初始化
'''
global data  # 保存用户数据
global last_param_list  # 保存上一次参数
global tmp_param_list  # 保存最新参数
global algorithm_type  # 取值1，2，3
global tree  # 存储误差绝对值和segment_tree
'''
定义误差常量
'''
EPSILON = 0.001


def main():
    dataset = pd.read_csv('input/iris2.csv')
    print(dataset.head())
    y = dataset['Label']
    X = dataset

    model = SVC(gamma='auto')
    scores = cross_val_score(model, X, y, cv=10)
    print(scores.mean())

    a = find_local_supreme(dataset, 'SVM', [29, 100])
    print(a)


# def find_local_supreme(dataset, alg_name, predicted_param):
#    '''
#    在神经网络给出结果的基础上，寻找局部极值，进一步优化超参数
#    Parameters:
#      dataset - 一个用户数据集，类型为pandas.DataFrame，每行一个样本，第一行是属性名称，第二行起是数据
#      alg_name - 待调参算法名称，String类型，'SVM','ElasticNet','GMM'中的一个
#      predicted_param - 神经网络的预测结果，一个列表，包含各个参数预测的值
#    Returns:
#      一个列表，包含各个参数的最终优化结果，要求参数次序和predicted_param相同
#    初始化全局变量
#    '''
#    if alg_name == 'GMM':
#        return predicted_param
#    global data#保存用户数据
#    data = dataset
#    global last_param_list#保存上一次参数
#    last_param_list = [-1000000 for x in range(len(predicted_param))]
#    global tmp_param_list#保存最新参数
#    tmp_param_list = predicted_param
#    global algorithm_type#取值1，2，3
#    if alg_name == 'SVM':
#        algorithm_type = 1
#    if alg_name == 'ElasticNet':
#        algorithm_type = 2
#    if alg_name == 'GMM':
#        algorithm_type = 3
#    global tree#存储误差绝对值和segment_tree
#    tree = [1000000 for x in range(4 * len(predicted_param))]
#    tree_init(1,len(predicted_param),1)
#    #begin running
#    func(1,len(predicted_param),1)
#    # return predicted_param
#    return tmp_param_list

def find_local_supreme(dataset, alg_name, predicted_param):
    '''
    在神经网络给出结果的基础上，寻找局部极值，进一步优化超参数
    Parameters:
      dataset - 一个用户数据集，类型为pandas.DataFrame，每行一个样本，第一行是属性名称，第二行起是数据
      alg_name - 待调参算法名称，String类型，'SVM','ElasticNet','GMM'中的一个
      predicted_param - 神经网络的预测结果，一个列表，包含各个参数预测的值
    Returns:
      一个列表，包含各个参数的最终优化结果，要求参数次序和predicted_param相同
    初始化全局变量
    '''
    if alg_name == 'GMM':
        return predicted_param
    global data  # 保存用户数据
    data = dataset
    global last_param_list  # 保存上一次参数
    last_param_list = [-1000000 for x in range(3)]
    global tmp_param_list  # 保存最新参数
    tmp_param_list = predicted_param
    global algorithm_type  # 取值1，2，3
    if alg_name == 'SVM':
        algorithm_type = 1
        tmp_param_list[0] = math.log2(tmp_param_list[0])
        tmp_param_list[1] = math.log2(tmp_param_list[1])
    if alg_name == 'ElasticNet':
        algorithm_type = 2
        tmp_param_list[0] = math.log2(tmp_param_list[0])
        tmp_param_list[1] = math.log2(tmp_param_list[1])
    if alg_name == 'GMM':
        algorithm_type = 3
    global tree  # 存储误差绝对值和segment_tree
    tree = [1000000 for x in range(4 * 2)]
    tree_init(1, 2, 1)
    # begin running
    func(1, 2, 1)
    # return predicted_param
    tmp_param_list[0] = pow(2, tmp_param_list[0])
    tmp_param_list[1] = pow(2, tmp_param_list[1])

    # print()
    # print('final params:')
    # print(tmp_param_list)
    # print('\n\n\n')

    return tmp_param_list


'''
建立测试环境
返回正确率
'''


def env():
    if algorithm_type == 1:
        return env1()
    if algorithm_type == 2:
        return env2()
    if algorithm_type == 3:
        return env3()
    return


'''
SVM
'''


def env1():
    params = {}
    names = knowledge.KnowledgePrepare.get_continuous_para_name('SVM')
    #    names = ["C", "gamma"]
    lenth = 2
    for i in range(lenth):
        params[names[i]] = pow(2, tmp_param_list[i])
    return svm_cross_validation(data, params)


'''
ElasticNet
'''


def env2():
    params = {}
    names = knowledge.KnowledgePrepare.get_param_name('ElasticNet')
    lenth = len(names)
    for i in range(lenth):
        params[names[i]] = pow(2, tmp_param_list[i])
    return elastic_net_cross_validation(data, params)


'''
GMM
'''


def env3():
    params = {}
    names = knowledge.KnowledgePrepare.get_param_name('GMM')
    lenth = len(names)
    for i in range(lenth):
        params[names[i]] = tmp_param_list[i]
    return gmm_score(data, params)


'''
多个参数调优
'''


def func(l, r, o):
    if l == r:
        func1(l)
        return
    if l + 1 == r:
        func2(l, r)
        return
    mid = int((l + r) / 2)
    while True:
        func(l, mid, 2 * o)
        if check(l, r, o) == True:
            break
        func(mid + 1, r, 2 * o + 1)
        if check(l, r, o) == True:
            break
    return


'''
两个参数调优
'''


def func2(l, r):
    hill_climb(l, r)
    global last_param_list
    global tmp_param_list
    x = abs(tmp_param_list[l - 1] - last_param_list[l - 1])
    y = abs(tmp_param_list[r - 1] - last_param_list[r - 1])
    update(1, len(tmp_param_list), 1, l, x)
    update(1, len(tmp_param_list), 1, r, y)
    last_param_list = tmp_param_list
    return


'''
单点set
'''


def update(l, r, o, index, value):
    if l == r:
        tree[o] = value
        return
    mid = int((l + r) / 2)
    if index <= mid and index >= l:
        update(l, mid, 2 * 0, index, value)
    if index > mid and index <= r:
        update(mid + 1, r, 2 * o + 1, index, value)
    tree[o] = tree[2 * o] + tree[2 * o + 1]
    return


'''
初始化树结构
'''


def tree_init(l, r, o):
    if l == r:
        tree[o] = 100
        return
    mid = int((l + r) / 2)
    tree_init(l, mid, 2 * o)
    tree_init(mid + 1, r, 2 * o + 1)
    tree[o] = tree[2 * o] + tree[2 * o + 1]
    return


'''
区间check
'''


def check(l, r, o):
    if tree[o] <= (r - l + 1) * EPSILON:
        return True
    return False


'''
爬山法
'''


def hill_climb(l, r):
    func1(l)
    func1(r)
    func1(l)
    func1(r)
    func1(l)
    func1(r)
    print('hill is over')
    print()
    return


'''
一元爬山法
'''


def func1(index):
    stride = 1
    x = tmp_param_list[index - 1]
    while stride >= 0.05:
        print("stride =", stride)
        tmp_param_list[index - 1] = x - stride
        if algorithm_type == 1:
            if tmp_param_list[index - 1] < -10:
                tmp_param_list[index - 1] = -10
        if algorithm_type == 2:
            if index == 1 and tmp_param_list[index - 1] < -10:
                tmp_param_list[index - 1] = -10
            if index == 2 and tmp_param_list[index - 1] < -6.6428:
                tmp_param_list[index - 1] = -6.6428
        a = env()
        # print()
        # print('stride')
        # print(stride)
        # print('abc')
        # print(a)
        tmp_param_list[index - 1] = x
        b = env()
        # print(b)
        tmp_param_list[index - 1] = x + stride
        if algorithm_type == 1:
            if tmp_param_list[index - 1] > 7:
                tmp_param_list[index - 1] = 7
        if algorithm_type == 2:
            if index == 1 and tmp_param_list[index - 1] > 7:
                tmp_param_list[index - 1] = 7
            if index == 2 and tmp_param_list[index - 1] > 0:
                tmp_param_list[index - 1] = 0
        c = env()
        # print(c)
        if a > b:
            x = x - stride
        elif c > b:
            x = x + stride
        else:
            stride = stride / 2
        tmp_param_list[index - 1] = x
        print('temp params')
        print(tmp_param_list)
        print()
    return


'''
构建临时func1
'''


def tmpfunc1(index):
    stride = 10
    x = tmp_param_list[index - 1]
    while stride >= 1:
        tmp_param_list[index - 1] = x - stride
        a = env()
        tmp_param_list[index - 1] = x
        b = env()
        tmp_param_list[index - 1] = x + stride
        c = env()
        if a > b:
            x = x - stride
        elif c > b:
            x = x + stride
        else:
            stride = stride / 2
    return


def svm_cross_validation(dataset, params):
    '''
    SVM交叉验证
    Parameters:
      待跑SVM的数据集，格式：pandas.DataFrame
    Returns:
      数据集10折交叉验证得分，使用默认的SVM超参数，评估标准：正确率
    '''
    y = dataset['Label']
    X = dataset

    model = SVC(**params)
    scores = cross_val_score(model, X, y, cv=10)

    ret = scores.mean()
    return ret


def elastic_net_cross_validation(dataset, params):
    '''
    ElasticNet交叉验证
    Parameters:
      待运行ElasticNet的数据集，格式：pandas.DataFrame
    Returns:
      数据集10折交叉验证得分，使用默认的ElasticNet超参数，评估标准：r2
    '''
    import copy
    data = copy.deepcopy(dataset)
    y = data.pop('Label')
    X = data

    model = ElasticNet(**params)
    model.fit(X, y)
    scores = cross_val_score(model, X, y, cv=10, scoring='r2')

    ret = scores.mean()
    return ret


def gmm_score(dataset, params):
    '''
    GMM聚类效果评估
    Parameters:
      待运行GMM的数据集，格式：pandas.DataFrame
    Returns:
      数据集聚类结果，使用已知的簇的个数，其他超参数使用默认值，评估标准：adjusted_rand_score
    '''
    import copy
    data = copy.deepcopy(dataset)
    y_true = data.pop('Label')
    X = data

    y_pred = GMM(**params).fit_predict(X)

    ret = adjusted_rand_score(y_true, y_pred)
    return ret


if __name__ == '__main__':
    main()