# 描述

本项目试图构造一个自动调参系统。



# 规范

## 0.运行

在项目根目录输入py 'XX/YYY.py' 来运行文件


## 1.命名规则

目录名（除database内文件夹外）使用小写字符
文件名使用驼峰式
函数名使用小写字母和下划线，一般为动词+名词


## 2.目录结构

- AutoParamTuning

    - README.txt
    - knowledge: 包含各个数据集特征和最优参数组成的新数据集
        - KnowledgePrepare.py: 对database中的数据集计算特征和最优参数，保存到下面三个csv文件
        - NetworkPrepare.py: 训练神经网络，保存到system/network中
        - Svm.csv
        - ElasticNet.csv
        - Gmm.csv

    - system: 包含调参器主体
        - AutoParamTuning.py: 客户端
        - FeatureCalc.py: 特征计算器
        - network: 训练好的神经网络
            - SvmModel.h5
            - ElasticNetMode.h5
            - GmmModel.h5
        - FurtherOpt.py: 使用三分法进一步优化
        - input: 存放待调参数据集的目录
            - ZZZ.csv
            - ...
        - output: 保存结果
            - Features.csv 输入数据集的特征
            - InitialResult.csv 神经网络给出的结果
            - FinalResult.csv 最终结果

    - evaluation: 评估调参结果的代码
        - KnowledgeEvaluate.py: 对knowledge中的数据集训练神经网络，并进行交叉验证，看看拟合优度
        - ResultEvaluate.py: 使用默认参数、网格搜索的参数、我们得到的参数分别在用户数据集运行待调参算法，生成下面的评估结果
        - EvaluationResult.csv: 评估结果
        - DatasetFilter.py: 筛选../database中的数据集，若表现过差，将该数据集移动到../database/_OffSpec中


**以下目录不在github该项目中上传，但应和AutoParamTuning放于同一目录下**
- database_init: 包含适用于三种算法的数据集
    - _OffSpec: 存放表现过差的数据集
        SVM
        GMM
        ElasticNet
    - SVM:
        - XXX.csv
        - YYY.csv
        - ...
    - ElasticNet
        - ...
    - GMM
        - ...
- database: 存放预处理过的数据集
    - SVM:
        - XXX.csv
        - ...
    - ElasticNet
        - ...
    - GMM
        - ...

## 3. 数据表示

- 所有外部数据集使用csv格式文件存储，第一行为属性名称，从第二行开始为数据文件主体不包含行号，从第一列开始就是数据
- 所有外部数据集都是已经经过预处理的，不包含字符串属性值，已经标准化过
- output中csv的格式为，首行为数据集名称，首列为特征名称或超参数名称，其余部分为数据
- 程序中，所有读入的数据集表示为pandas.Dataframe类型
- 所有文件路径从本项目根目录开始写


## 4. 其他规约
- 0 * log2(0) = 0