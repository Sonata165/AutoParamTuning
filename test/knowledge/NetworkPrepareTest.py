import os
from knowledge.NetworkPrepare import read_data, train_test_nn_for_model
from knowledge.KnowledgePrepare import get_feature
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    print("设置目录")
    os.chdir("../../")
    print("生成参数")
    get_feature("SVM")
    print("准备数据")
    x, y = read_data("SVM")
    print("搭建网络开始训练")
    train_test_nn_for_model("SVM", 10, x, y, input_shape=(36,), output_dim=4, save_path='system/network/')
