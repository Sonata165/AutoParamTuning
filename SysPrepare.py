import sys
import time
from multiprocessing import freeze_support

sys.path.append('system')
sys.path.append('knowledge')

import KnowledgePrepare
import NetworkPrepare


def main():
    freeze_support()
    print('生成训练网络用数据集\n1.SVM\t2.ElasticNet\t3.GMM\t4.all')
    choice = int(input())
    if choice == 1:
        model_name = "SVM"
    elif choice == 2:
        model_name = 'ElasticNet'
    elif choice == 3:
        model_name = 'GMM'
    elif choice == 4:
        pass

    KnowledgePrepare.get_feature(model_name)
    print('生成完成！')

    time.sleep(2)

    print('训练神经网络\n1.SVM\t2.ElasticNet\t3.GMM\t4.all')
    choice = int(input())
    if choice == 1:
        x, y = NetworkPrepare.read_data(model_name)
        NetworkPrepare.train_test_nn_for_model(model_name, 10, x, y, input_shape=(36,), 
            output_dim=4, save_path='system/network/')
    elif choice == 2:
        pass
        #TODO
    elif choice == 3:
        pass
        #TODO
    elif choice == 4:
        pass
        #TODO

    print('训练完成')

if __name__ == '__main__':
    main()
