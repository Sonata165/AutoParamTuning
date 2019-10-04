'''
Author: 欧龙燊
本文件下函数都和数据集的特征计算有关
'''

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DISCRETE_BOUND = 10    # 当一个属性的取值小于等于它时，该属性被视作离散的
TOT_CORR_BOUND = 10000 # 当总相关性大于它时，将被记作它
FEATURENAMES = [ # 表中是“共有特征”名称
    'AttributeNum',
    'DiscreteRatio',
    'JointInf',
    'Size',
    'Totcorr',

    'Mean1',
    'Mean2',
    'Mean3',
    'Mean4',

    'Std1',
    'Std2',
    'Std3',
    'Std4',

    'Median1',
    'Median2',
    'Median3',
    'Median4',

    'Min1',
    'Min2',
    'Min3',
    'Min4',

    'Max1',
    'Max2',
    'Max3',
    'Max4',

    'Skew1',
    'Skew2',
    'Skew3',
    'Skew4',

    'Entropy1',
    'Entropy2',
    'Entropy3',
    'Entropy4',

    'PCAJointInf'
]

def calculate_features(alg_name, dataset):
    '''
    计算特征
    Parameters:
      alg_name - 字符串，待调参算法名称
      dataset - 一个待计算特征的数据集
    Returns:
      列表，所有计算好的特征
    '''
    # 先计算共有的特征
    features = calculate_common_features(dataset)

    # 然后计算特有的特征
    if alg_name == 'SVM':
        # 计算标签信息熵
        label_entropy = calculate_label_entropy(dataset)
        # 计算标签数目
        label_num = calculate_label_num(dataset)
        features.append(label_entropy)
        features.append(label_num)
        return features
    elif alg_name == 'ElasticNet':
        # 计算标签信息熵
        label_entropy = calculate_label_entropy(dataset)
        features.append(label_entropy)
        return features    
    elif alg_name == 'GMM':
        pass


def calculate_common_features(dataset):
    '''
    计算所有待调参算法的数据集的“共有特征”
    Parameters:
      dataset - 一个待计算特征的数据集
    Returns:
      列表，所有计算好的共有特征
    '''
    attribute_num = calculate_attribute_num(dataset)
    discrete_ratio = calculate_discrete_ratio(dataset)
    # entropy = calculate_entropy(dataset)
    joint_inf = calculate_joint_inf(dataset)
    size = calculate_size(dataset)
    totcorr = calculate_totcorr(dataset)

    pca = PCA(4)
    dataset = pd.DataFrame(pca.fit_transform(dataset))
    stat = dataset.describe().transpose()
    print(dataset.head())
    print(stat)

    mean1 = stat['std'][0]
    mean2 = stat['std'][1]
    mean3 = stat['std'][2]
    mean4 = stat['std'][3]

    std1 = stat['std'][0]
    std2 = stat['std'][1]
    std3 = stat['std'][2]
    std4 = stat['std'][3]

    median1 = np.median(dataset[0])
    median2 = np.median(dataset[1])
    median3 = np.median(dataset[2])
    median4 = np.median(dataset[3])

    min1 = stat['min'][0]
    min2 = stat['min'][1]
    min3 = stat['min'][2]
    min4 = stat['min'][3]

    max1 = stat['max'][0]
    max2 = stat['max'][1]
    max3 = stat['max'][2]
    max4 = stat['max'][3]

    skew1 = dataset[0].skew()
    skew2 = dataset[1].skew()
    skew3 = dataset[2].skew()
    skew4 = dataset[3].skew()

    entropy1 = calculate_entropy(dataset[0])
    entropy2 = calculate_entropy(dataset[1])
    entropy3 = calculate_entropy(dataset[2])
    entropy4 = calculate_entropy(dataset[3])

    pca_joint_inf = calculate_joint_inf(dataset)

    ret = [attribute_num, discrete_ratio, joint_inf, size, totcorr,
            mean1, mean2, mean3, mean4, std1, std2, std3, std4,
            median1, median2, median3, median4, min1, min2, min3, min4,
            max1, max2, max3, max4, skew1, skew2, skew3, skew4,
            entropy1, entropy2, entropy3, entropy4, pca_joint_inf]

    return ret


def get_feature_name(alg_name):
    '''
    获取计算过的特征的名称
    '''
    ans = FEATURENAMES

    if alg_name == 'SVM':
        ans.append('LabelEntropy')
        ans.append('LabelNum')
    elif alg_name == 'ElasticNet':
        ans.append('LabelEntropy')
    elif alg_name == 'GMM':
        pass
    else:
        die
    return FEATURENAMES


def calculate_size(dataset):
    '''
    计算数据集大小
    Parameters:
      dataset - 待计算的数据集
    Returns:
      该数据集的大小（样本数目）
    '''
    return len(dataset)


def calculate_discrete_ratio(dataset):
    """
    计算数据集的离散属性比率
    Parameters:
      dataset - 待计算的数据集
    Returns:
      该数据集的离散属性比率
    """
    discrete_count = 0
    for i in dataset.columns:
        if len(dataset[i].value_counts()) == 2:
            discrete_count += 1
    return discrete_count / dataset.columns.size


def calculate_attribute_num(dataset):
    """
    计算数据集属性个数
    Parameters:
      dataset - 待计算的数据集
    Returns:
      该数据集的属性个数
    """
    return dataset.columns.size


def calculate_label_entropy(dataset):
    """
    计算指定数据集的信息熵(只看标签一栏) -Sigma(P(yi)log_2(P(yi)))  加和次数等于标签数目
    Parameters:
      dataset - 待计算的数据集
    Returns:
      该数据集的信息熵
    """
    entropy = 0
    dic = dataset['Label'].value_counts()        # 它保存着不同标签的出现次数
    n = len(dataset)
    for i in dic:
        pr = i / n
        entropy -= pr * np.log2(pr)
    return entropy


def calculate_entropy(array):
    '''

    '''
    entropy = 0
    dic = array.value_counts()
    n = len(array)
    for i in dic:
        pr = i / n
        entropy -= pr * np.log2(pr)
    return entropy


def calculate_joint_inf(dataset):
    """
    计算指定数据集的联合信息熵 -Sigma(P(x1,x2)log_2(P(x1,x2)))
    Parameters:
      dataset - 待计算的数据集
    Returns:
      该数据集的联合信息熵
    """
    lines = []
    for index, row in dataset.iterrows():
        line_list = list(row)
        lines.append(str(line_list))
    lines = pd.Series(lines)
    dic = lines.value_counts()        # 它保存着每行的toString和不同行的出现次数
    b = len(dic)                    # b表示有多少种不同的行
    joint_inf = 0
    n = len(dataset)
    for i in dic:
        pr = i / n                    # 该行出现的概率
        joint_inf -= pr * np.log2(pr)
    return joint_inf


def calculate_totcorr(dataset):
    """
    对一个数据集计算总相关性
    TODO Buggy
    Parameters:
      dataset - 待计算的数据集
    Returns:
      该数据集的总相关性
    """
    lines = []
    for index, row in dataset.iterrows():
        line_list = list(row)
        lines.append(str(line_list))
    lines = pd.Series(lines)
    dic = lines.value_counts()        # 它保存着每行的toString和不同行的出现次数
    # 建立一个<line, 次数>的dic
    dic = dict(dic)
    # 对每一列都建立一个字典
    dic_of_column = []
    for index in dataset.columns:
        dic_of_column.append(dict(dataset[index].value_counts()))
    # 对于每行，对于每个元素，计算概率乘积，然后按规则计算
    n = len(dataset) # n是数据集行数
    tot_corr = 0
    for line_str in dic:
        line_tmp = line_str[1:-1].split(', ')
        line = []
        for i in line_tmp:
            line.append(np.float64(i))
        product = np.float64(1.0)
        for i in range(0, len(line)):
            product *= dic_of_column[i][line[i]] / n
        pr = dic[line_str] / n         # 该行出现的概率
        tot_corr += pr * np.log2(pr/product)
    return min(tot_corr, TOT_CORR_BOUND)


def calculate_label_num(dataset):
    '''
    计算标签取值的数目.
    Parameters:
      dataset - 待计算特征的数据集
    Retures:
      该数据集中标签取值的数目
    '''
    return len(dataset['Label'].value_counts())

def main():
    print("")


if __name__ == '__main__':
    main()