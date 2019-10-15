import os
from knowledge.NetworkPrepare import read_data, train_test_nn_for_model
from knowledge.KnowledgePrepare import get_feature
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    modelName = "SVM"
    print("设置目录")
    os.chdir("../../")
    print("生成参数")
    accuracy = 1
    r_gamma = [i / accuracy for i in range(0 * accuracy, 2 * accuracy, 1)]
    r_C = [i / accuracy for i in range(1 * accuracy, 2 * accuracy, 1)]
    param_grid = {
        'kernel': ['rbf'],
        'C': r_C,
        'gamma': r_gamma}
    get_feature(modelName)

    print("准备神经网络训练数据")
    x, y = read_data(modelName)
    print("搭建网络开始训练")
    train_test_nn_for_model(modelName, 10, x, y, input_shape=(36,), output_dim=4, save_path='system/network/')
    """
    目前，SVM特征数为36
        ElasticNet为35
        GMM为34
    """
