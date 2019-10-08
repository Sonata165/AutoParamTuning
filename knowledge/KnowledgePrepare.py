"""
计算database目录下所有数据集的特征和最优参数（网格搜索）
database目录结构如README中所示
计算特征时，可以调用system/FeatureCalc.py中的calculate_features(dataset)，
其中dataset是一个读入的数据集（用read_csv函数读入，类型为DataFrame），返回值是所有特征的列表
计算最优参数时，可以参考原来的old/compute_label.py
全部计算完后，将结果保存在如README所示knowledge下三个csv文件中
保存格式如下：
	需要把最优参数放在后面的列，特征放在前面的列
	第一行是特征名和参数名，从第二行开始是数据
	第一列不是编号，从第一列开始就是数据
"""
import sys
sys.path.append('system')
from FeatureCalc import *
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet
from sklearn.mixture import GaussianMixture
import numpy as np

"""
def main():
    get_feature('SVM', database_dir='../database/', param_g={'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [1, 10, 50, 10]})
"""


def get_feature(modelName, database_dir='../database/', param_g=None):
    """
    获取特征-标签数据
    返回最优参数的同时保存数据，格式为：
          feature1  feature2 ... best_param(label)
    name:
    data:
    :param modelName: 要优化的模型，字符串，区分大小写，例：'SVM','ElasticNet','GaussianMixture
    :param param_g: 网格搜索范围， 默认值为None，并使用内置的搜索范围，格式如下：
                    例:SVM
                    {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                     'C': [1, 10, 50, 10],
                     'gamma': r_gamma}
    :return: 所有特征的列表
    """

    dataset_dir = database_dir + modelName + "/"
    dir_list = os.listdir(dataset_dir)
    feature = None
    gs_list = []

    for p in dir_list:
        path = dataset_dir + p
        print("path: " + path)
        # 读取数据
        data = pd.read_csv(path)
        # print(data)
        data = np.array(data)
        x_data = data[:, 0:-1]
        y_data = (data[:, -1])

        # print(x_data)

        # 网格搜索标签
        if modelName == "SVM":
            # 搜索精度
            accuracy = 1
            # 生成搜索范围
            r_gamma = [i / accuracy for i in range(0*accuracy, 2 * accuracy, 1)]
            r_C = [i / accuracy for i in range(1 * accuracy, 2 * accuracy, 1)]
            if param_g is None:
                # param_grid = {'kernel': ['linear'], 'C': r_C, 'gamma':1}
                param_grid = {'kernel': ['linear'], 'C': r_C, 'gamma': r_gamma}
            else:
                param_grid = param_g
            model = SVC()
            print("开始网格搜索")
            gs = GridSearchCV(model, param_grid=param_grid, refit=True, cv=5).fit(x_data, y_data)
            print("网格搜索完成")
        elif modelName == 'ElasticNet':
            # 搜索精度
            accuracy = 1
            # 生成搜索范围
            r_alpha = [i / accuracy for i in range(0 * accuracy, 2 * accuracy, 1)]
            r_l1 = [i / accuracy for i in range(0 * accuracy, 1 * accuracy, 1)]
            param_grid = {'alpha': r_alpha, 'l1_ratio': r_l1}
            model = ElasticNet()
            print("开始网格搜索")
            gs = GridSearchCV(model, param_grid=param_grid, cv=5).fit(x_data, y_data)
            print("网格搜索完成")
        elif modelName == 'GMM':
            param_grid = {'n_components': [1, 2, 3], 'covariance_type': ['full', 'tied', 'diag', 'spherical']}
            model = GaussianMixture()
            print("开始网格搜索")
            gs = GridSearchCV(model, param_grid=param_grid, cv=5).fit(x_data, y_data)
            print("网格搜索完成")
        else:
            print("模型名称输入错误！")
            return None
        gs_list.append(gs)

        # 输出网格搜索结果
        print("best_params:\n" + str(gs.best_params_))
        print('best_score:\n' + str(gs.best_score_))

    for p in dir_list:
        path = os.path.join(dataset_dir, p)
        print("path: " + path)
        # 读取数据
        data = pd.read_csv(path)
        # 计算数据集特征并合并存储
        temp = np.array(calculate_features(modelName, data))
        if feature is None:
            feature = temp.reshape(temp.shape[0], 1)
        else:
            feature = np.concatenate((feature, temp.reshape(temp.shape[0], 1)), axis=1)

    # 添加最优参数行
    param_name = [k for k in gs_list[0].best_params_]
    args = get_param_name(modelName)
    if modelName == "SVM":
        k = args[-1]
        args[-1] = k + '0'
        args.append(k + '1')
        args.append(k + '2')
        args.append(k + '3')
        param_name = args[-6:]
    elif modelName == "GMM":
        k = args[-2]
        args[-2] = k + '0'
        args.insert(-1, k + '1')
        args.insert(-1, k + '2')
        args.insert(-1, k + '3')
        param_name = args[-5:]
    feature = pd.DataFrame(feature, columns=[p for p in dir_list])
    param = []
    for g_dataset in gs_list:
        # example: g_dataset={""C":0,"gamma":0,kernel":"rbf"}
        col = []
        for k in g_dataset.best_params_.values():
            # one-hot encode: ['linear', 'rbf', 'poly', 'sigmoid'],['full', ' tied', 'diag', 'spherical']
            code = [0, 0, 0, 0]
            if k == 'linear' or k == 'full':
                code[0] = 1
                col += code
            elif k == 'rbf' or k == 'tied':
                code[1] = 1
                col += code
            elif k == 'poly' or k == 'diag':
                code[2] = 1
                col += code
            elif k == 'sigmoid'or k == 'spherical':
                code[3] = 1
                col += code
            else:
                col.append(k)
        col = np.array(col).transpose()
        # 添加列
        param.append(col)

    param = pd.DataFrame(np.array(param).transpose(), index=[s for s in args], columns=[p for p in dir_list])
    # 合并
    res = pd.concat([feature, param], axis=0, ignore_index=True)
    # 添加FeatureName列
    name = get_feature_name(modelName) + param_name
    name_col = pd.DataFrame(np.array(name).reshape((len(name), 1)))
    res = pd.concat([name_col, res], axis=1, ignore_index=True)
    # 保存特征和超参数
    res.columns = ["FeatureName"] + [p for p in dir_list]
    res = pd.DataFrame(res.values.T, index=res.columns, columns=res.index)
    res.to_csv("knowledge/" + modelName + ".csv", index=False, header=None)
    return np.array(feature).tolist()


def get_param_name(alg_name):
    """
    返回待调的超参数的名称
    Parameters:
      alg_name - String类型，待调参算法名称，'SVM','ElasticNet','GMM'中的一个
    Returns:
      一个列表，包含该算法要调的参数的名称。
      要求列表大小和神经网络输出数目总数保持一致，顺序也和神经网络输出保持一致
    """
    if alg_name == "SVM":
        return ["C", "gamma", "kernel"]
    elif alg_name == "ElasticNet":
        # TODO:确保顺序正确性
        return ["alpha", "l1_ratio"]
    elif alg_name == "GMM":
        return ["n_components", "covariance_type"]
    return None


if __name__ == '__main__':
    main()
