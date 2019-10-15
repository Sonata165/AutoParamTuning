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
    print('生成训练网络用数据集\n1.SVM\t2.ElasticNet\t3.GMM\t4.all')
    choice = int(input())

    if choice == 1:
        modelName = "SVM"
        print("生成参数")
        accuracy = 1
        r_gamma = [i / accuracy for i in range(0 * accuracy, 2 * accuracy, 1)]
        r_C = [i / accuracy for i in range(1 * accuracy, 2 * accuracy, 1)]
        param_grid = {
            'kernel': ['rbf'],
            'C': r_C,
            'gamma': r_gamma}
        get_feature(modelName)
    elif choice == 2:
        model_name = 'ElasticNet'
    elif choice == 3:
        model_name = 'GMM'
    elif choice == 4:
        pass

    modelName = 'SVM'
    print('生成完成！')

    time.sleep(1)
    print("准备神经网络训练数据")
    x, y = read_data(modelName)

    print('训练神经网络\n1.SVM\t2.ElasticNet\t3.GMM\t4.all')
    choice = int(input())
    if choice == 1:
        train_test_nn_for_model(modelName, 10, x, y, input_shape=(36,), output_dim=4, save_path='system/network/')
    elif choice == 2:
        pass
        #TODO
    elif choice == 3:
        pass
        #TODO
    elif choice == 4:
        pass
        #TODO
    print('\n训练完成')

if __name__ == '__main__':
    main()
