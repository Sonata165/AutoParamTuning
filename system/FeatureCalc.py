'''
Author: 欧龙燊
本文件下函数都和数据集的特征计算有关
'''
import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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

STUFFING_RATE = 1
PADDING_TO = 30
HIDDEN_LAYER_SIZE = 15
EPOCHS = 2000
C_type = [np.dtype('float16'), np.dtype('float32'), np.dtype('float64')]
D_type = [np.dtype('int8'), np.dtype('int16'), np.dtype('int32'), np.dtype('int64')]

def calculate_features_nn(dataset):
    '''
    计算特征
    Parameters:
      dataset - 一个待计算特征的数据集
    Returns:
      列表，通过NN ENCODING计算好的特征
    '''
    if dataset.shape[1] > PADDING_TO:
        pca = PCA(PADDING_TO-1)
        res = pca.fit_transform(dataset)
        dataset = pd.DataFrame(pca.fit_transform(dataset))
    random_data = generate_random_data(dataset, STUFFING_RATE)
    padded_data = padding(PADDING_TO, dataset, random_data)
    ret = nn_score(padded_data)
    return ret

def generate_random_data(dataset, stuffing_rate):
    '''
    Parameters:
      dataset - 原数据集
      stuffing_rate - 随机取点的数目和原数据集样本数目的比例
    Returns:
      新的随机点的数据集
    '''
    ret = pd.DataFrame()
    for index in dataset.columns:
        cnt = dataset[index].value_counts()
        tmp = cnt.sort_index()

        # 确定新数据的一列的范围
        # [a,b,...,c,d]
        a = tmp.index[0]
        b = tmp.index[1]
        c = tmp.index[-2]
        d = tmp.index[-1]

        # 生成数据
        # if index in C_or_D.iris_c:
        if dataset[index].dtype in C_type:
            new_list = random_list('C', dataset[index].count() * STUFFING_RATE, a, b, c, d)
        elif dataset[index].dtype in D_type:
            new_list = random_list('D', dataset[index].count() * STUFFING_RATE, a, b, c, d)
        else:
            die
        new_s = pd.Series(new_list, name=index)
        ret[index] = new_s
    return ret

def random_list(type, size, a, b, c, d):
    '''
    只由Generate_random_data函数调用
    根据提供范围限制和类型，新生成一个list. 
    若一列数据的取值范围是[a,b,...,c,d], 
    则新列的范围是[a-(b-a), d+(d-c)]. 
    Parameters:
      type - 数据种类，取值为'D'（离散）或'C'（连续）
      size - 新list的大小
      a, b, c, d - 范围限制
    Returns:
      生成的随机数列表。生成规则：均匀分布
    '''
    min_v = a-(b-a)
    max_v = d+(d-c)
    if type == 'C':
        ret = np.random.uniform(min_v, max_v, size)
    elif type == 'D':
        ret = np.random.randint(min_v, max_v+1, size)
    return ret

def padding(dim, df1, df2):
    '''
    将两个数据集拼接然后0-padding到指定维度. 维度数目包括原标签列以及新增的标签列.
    Parameters:
      dim - 指定的padding后的维度
      df1 - 第一个数据集，要求shape[1] <= PADDING_TO - 1
      df2 - 第二个数据集，要求shape[1] <= PADDING_TO - 1
    '''
    dataset = merge_dataset(df1, df2)
    print('sha;e')
    print(dataset.shape[1])

    new_col_num = dim - dataset.shape[1]
    label = dataset.pop('Original')
    for i in range(0, new_col_num):
        name = 'Pad_' + str(i)
        dataset[name] = 0
    dataset['Original'] = label
    return dataset

def merge_dataset(df1, df2):
    '''
    只能由padding函数调用. 
    将原数据集和随机取点生成的数据集合并. 
    Parameters:
      df1 - 原数据集
      df2 - 随机取点的数据集
    Returns:
      将df2接在df1后面，新增一个标签：是不是原数据集的样本，然后打乱
    '''
    tmp1 = df1.copy()
    tmp2 = df2.copy()
    tmp1['Original'] = 1
    tmp2['Original'] = 0

    ret = tmp1.append(tmp2)
    ret = ret.sample(frac=1)
    ret = ret.reset_index(drop=True)
    return ret

def nn_score(dataset):
    '''
    看看神经网络的表现
    '''
    X = dataset.copy()
    y = X.pop('Original')
    # y = pd.get_dummies(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    model = build_model(X_train)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, 
            mode='auto', restore_best_weights=True)

    history = model.fit(X_train, y_train,
            epochs=EPOCHS, verbose=0, validation_split=0.2,
            batch_size=32, callbacks=[early_stop])

    score = model.evaluate(X_test, y_test, batch_size=32)
    print(model.metrics_names)
    print("NN score:", score)

    hist = pd.DataFrame(history.history)
    # print(hist)
    # plot_history(history)

    global a
    a = model.get_weights()
    print(len(a))
    # for i in range(0, len(a)):
    #     print(a[i].size)
    res = np.hstack((a[0].flatten(), a[1], a[2].flatten(), a[3]))
    return list(res)

def build_model(X_train):
    '''
    只由nn_score调用
    建立一个二分类神经网络模型
    Parameters:
      X_train - 神经网络训练集
    Returns:
      建立的网络
    '''
    model = Sequential()
    # model.add(Dense(16, input_dim=len(X_train.keys()), activation='relu'))
    model.add(Dense(HIDDEN_LAYER_SIZE, input_dim=len(X_train.keys()), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model

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

    return features


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
    # print(dataset.head())
    # print(stat)

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


def get_feature_name():
    '''
    获取计算过的特征的名称
    '''
    ret = []
    for i in range(0, 466):
        ret.append("F_"+str(i))
    return ret


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