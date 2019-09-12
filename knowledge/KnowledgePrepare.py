'''
计算database目录下所有数据集的特征和最优参数（网格搜索）
database目录结构如README中所示
计算特征时，可以调用system/FeatureCalc.py中的calculate_features(dataset)，
其中dataset是一个读入的数据集（用read_csv函数读入，类型为DataFrame），返回值是所有特征的列表
计算最优参数时，可以参考原来的old/compute_label.py
全部计算完后，将结果保存在如README所示knowledge下三个csv文件中
	需要把最优参数放在后面的列，特征放在前面的列
	第一行是特征名和参数名，从第二行开始是数据
	第一列不是编号，从第一列开始就是数据
'''

from system.FeatureCalc import calculate_features
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet
from sklearn.mixture import GaussianMixture
import numpy as np


def get_feature(modelName, param_g=None):
    """
    获取特征-标签数据
    返回最优参数的同时保存数据，格式为：
          feature1  feature2 ... best_param(label)
    name:
    data:
    :param modelName: 要优化的模型，字符串，区分大小写，例：'SVM','ElasticNet','GaussianMixture
    :param param_g: 网格搜索范围， 默认值为None，并使用内置的搜索范围，格式如下：
                    例:SVM
                    {'kernel': ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed'], 'C': [1, 10, 50, 10]}
    :return: 所有特征的列表
    """
    # 搜索精度
    accuracy = 1000
    database_dir = '../database/' + modelName
    dir_list = os.listdir(database_dir)
    feature = []
    for p in dir_list:
        path = os.path.join(database_dir, p)
        print("path: " + path)
        # 读取数据
        data = pd.read_csv(path)
        # 计算特征
        feature.append(calculate_features(data))
        # 保存特征
        pd.DataFrame(feature).to_csv(p + '_feature.csv')
        # TODO:网格搜索数据准备
        x_data = np.array(data)[, :-1]
        y_data = np.array(data)[, -1:]
        # 网格搜索标签
        if modelName == "SVM":
            # 生成搜索范围
            r = [i / accuracy for i in range(-10 * accuracy, 10 * accuracy, 1)]
            if param_g is None:
                param_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed'], 'C': r,
                              'gamma': r}
            else:
                param_grid = param_g
            model = SVC()
            gs = GridSearchCV(model, param_grid=param_grid, refit=True, cv=5).fit(x_data, y_data)


        elif modelName == 'ElasticNet':
            param_grid = {'alpha': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, 0.01, 0.05, 0.1, 0.5, 1]}
            model = ElasticNet()
            gs = GridSearchCV(model, param_grid=param_grid, refit=True, cv=5).fit(x_data, y_data)

        elif modelName == 'GMM':
            param_grid = {'n_components': [1, 2, 3], 'covariance': ['full', 'tied', 'diag', 'spherical']}
            model = GaussianMixture()
            gs = GridSearchCV(model, param_grid=param_grid, refit=True, cv=5).fit(x_data, y_data)

        else:
            print("模型名称输入错误！")
            return None
        # 输出网格搜索结果
        print("best_params:\n" + str(gs.best_params_))
        print('best_score:\n' + str(gs.best_score_))

        path = os.path.join('/' + modelName + '/', p)
        # TODO:处理结果并保存

        # TODO:确定feature的类型并转为list
        return feature


def get_param_name(alg_name):
    '''
    返回待调的超参数的名称
    Parameters:
      alg_name - String类型，待调参算法名称，'SVM','ElasticNet','GMM'中的一个
    Returns:
      一个列表，包含该算法要调的参数的名称。
      要求列表大小和神经网络输出数目总数保持一致，顺序也和神经网络输出保持一致
    '''
    return ['a']
