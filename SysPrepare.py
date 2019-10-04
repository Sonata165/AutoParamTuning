import sys
import time

sys.path.append('system')
sys.path.append('knowledge')

import KnowledgePrepare
import NetworkPrepare

print('生成训练网络用数据集\n1.SVM\t2.ElasticNet\t3.GMM\t4.all')
choice = int(input())
if choice == 1:
    #TODO
elif choice == 2:
    #TODO
elif choice == 3:
    #TODO
elif choice == 4:
    #TODO
print('生成完成！')

time.sleep(2)

print('训练神经网络\n1.SVM\t2.ElasticNet\t3.GMM\t4.all')
choice = int(input())
if choice == 1:
    #TODO
elif choice == 2:
    #TODO
elif choice == 3:
    #TODO
elif choice == 4:
    #TODO
print('训练完成')
