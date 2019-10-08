import os
import sys

sys.path.append('knowledge')
sys.path.append('system')

from KnowledgePrepare import get_feature
from NetworkPrepare import read_data, train_test_nn_for_model

from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    modelName = "SVM"
    # print("设置目录")
    # os.chdir("../../")
    print("生成参数")
    get_feature(modelName)
    print("准备数据")
    x, y = read_data(modelName)
    print("搭建网络开始训练")
    train_test_nn_for_model(modelName, 10, x, y, input_shape=(36,), output_dim=4, save_path='system/network/')
    """
    目前，SVM特征数为36
        ElasticNet为35
        GMM为34
    """
