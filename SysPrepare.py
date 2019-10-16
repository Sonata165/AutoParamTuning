"""
目前，SVM特征数为36
    ElasticNet为35
    GMM为34
"""
import os
import sys
import numpy as np
import time
from multiprocessing import freeze_support

from knowledge import NetworkPrepare
from knowledge.KnowledgePrepare import get_feature
from knowledge.NetworkPrepare import read_data, train_test_nn_for_model


def main():
    freeze_support()
    print('选择模型\n1.SVM\t2.ElasticNet\t3.GMM')
    choice = int(input())
    if choice == 1:
        modelName = "SVM"
    elif choice == 2:
        modelName = 'ElasticNet'
    elif choice == 3:
        modelName = 'GMM'
    elif choice == 4:
        pass

    print("计算特征和标签吗？ y/n")
    choice = input('> ')
    # accuracy = 1
    # r_gamma = [i / accuracy for i in range(0 * accuracy, 2 * accuracy, 1)]
    # r_C = [i / accuracy for i in range(1 * accuracy, 2 * accuracy, 1)]
    # param_grid = {
    #     'kernel': ['rbf'],
    #     'C': r_C,
    #     'gamma': r_gamma}
    if choice == 'y':
        get_feature(modelName)
    else:
        pass
    print('特征和标签计算完成！')

    time.sleep(1)
    print("准备神经网络训练数据")
    x, y = read_data(modelName)

    print('训练神经网络')
    units = None
    if modelName == "SVM":
        units = 36
    elif modelName == "ElasticNet":
        units = 35
    elif modelName == "GMM":
        units = 34
    train_test_nn_for_model(modelName, int(1e4), x, y, input_shape=(units,), output_dim=4, save_path='system/network/')
    print('\n训练完成')

if __name__ == '__main__':
    main()
