import pandas as pd
import os
dataset_dir = '../database/GMM/'
dir_list = os.listdir(dataset_dir)
for p in dir_list:
    path = dataset_dir + p
    data = pd.read_csv(path)
    data.pop('Label')
    data.to_csv(path)
