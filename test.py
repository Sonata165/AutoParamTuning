from knowledge.KnowledgePrepare import *
import os
import pandas as pd

d = pd.read_csv('evaluation/Result_SVM.csv')
d = d.T
d.to_csv('evaluation/Result_SVM.csv')
