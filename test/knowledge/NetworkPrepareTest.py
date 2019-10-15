import os
from knowledge.NetworkPrepare import read_data, train_test_nn_for_model
from knowledge.KnowledgePrepare import get_feature
import numpy as np
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    print("设置目录")
    os.chdir("../../")
    namelist = ['SVM', 'ElasticNet', 'GMM']
    for modelName in namelist:

        # print("生成参数")
        # get_feature(modelName)
        print("准备数据")
        x, y = read_data(modelName)
        print("搭建网络开始训练")
        units = None
        if modelName == "SVM":
            units = 36
        elif modelName == "ElasticNet":
            units = 35
        elif modelName == "GMM":
            units = 34
        train_test_nn_for_model(modelName, int(1e4), x, y, input_shape=(units,), output_dim=4, save_path='system/network/')
        """ 
        目前，SVM特征数为36
            ElasticNet为35
            GMM为34
        """
