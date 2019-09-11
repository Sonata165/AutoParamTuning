# 描述

本项目试图构造一个自动调参系统。



# 规范

## 1.命名规则

目录名（除database内文件夹外）使用小写字符
文件名使用驼峰式
函数名使用小写字母和下划线，一般为动词+名词


## 2.目录结构

> README.md
> knowledge: 包含各个数据集特征和最优参数组成的新数据集
> > KnowledgePrepare.py: 对database中的数据集计算特征和最优参数，保存到下面三个csv文件
> > NetworkPrepare.py: 训练神经网络，保存到system/network中
> > Svm.csv
> > ElasticNet.csv
> > Gmm.csv
> system: 包含调参器主体
>>AutoParamTuning.py: 客户端
>>FeatureCalc.py: 特征计算器
>>network: 训练好的神经网络
>>>SvmModel.h5
>>>ElasticNetMode.h5
>>>GmmModel.h5
>>FurtherOpt.py: 使用三分法进一步优化
>>input: 存放待调参数据集的目录
>>>ZZZ.csv
>>>...
>>output: 保存结果
>>>AAA.csv 结果使用csv保存，便于处理
>evaluation: 评估调参结果的代码
>>KnowledgeEvaluate.py: 对knowledge中的数据集训练神经网络，并进行交叉验证，看看拟合优度
>>ResultEvaluate.py: 使用默认参数、网格搜索的参数、我们得到的参数分别行，  生成下面的评估结果
>>EvaluationResult.csv: 评估结果


**以下目录不在github该项目中上传**
>database: 包含适用于三种算法的数据集
>>	SVM:
>>>		XXX.csv
>>>		YYY.csv
>>>		...
>>	ElasticNet
>>>		...
>>	GMM
>>>		...

