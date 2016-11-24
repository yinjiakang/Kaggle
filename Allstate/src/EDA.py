import warnings

warnings.filterwarnings('ignore')

import numpy as py
import pandas as pd
import IPython

dataset = pd.read_csv("../data/train.csv")

dataset_test = pd.read_csv("../data/test.csv")

ID = dataset_test['id']

dataset_test.drop('id', axis = 1, inplace = True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print (dataset.head(5))

IPython.embed()
