import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from collections import Counter
from TrainingModel import *


config = {
    'round' : 200,
    'random_seed' : 1218,
    'fold' : 5
}

xgb_param = {
    'booster': 'gbtree',
    'objective' : 'multi:softmax',
    'num_class' : 4,
    'early_stopping_rounds':100,

    'max_depth' : 6,
    'eta': 0.1,
    'gamma' : 0.1,
    'min_child_weight':3,

    'subsample':0.7,    

    'seed': config['random_seed'],
    'nthread': 3,
}

"""
train_test = pd.read_csv('../data/input/train_test.csv')
card = pd.read_csv('../data/input/cardInfo.csv')
train_test_score = pd.read_csv('../data/input/train_test_score.csv')
train_test_w2v_feature = pd.read_csv('../data/input/w2v_features.csv')

train_test = pd.merge(train_test, card, how= 'left', on = 'ID')
train_test = pd.merge(train_test, train_test_score, how = 'left', on = 'ID')
train_test = pd.merge(train_test, train_test_w2v_feature, how = 'left', on = 'ID')
"""

train_test = pd.read_csv('../data/input/train_test0213.csv')
card = pd.read_csv('../data/input/cardInfo0213.csv')
train_test_score = pd.read_csv('../data/input/train_test_score0213.csv')
#train_test_w2v_feature = pd.read_csv('../data/input/w2v_features0213.csv')

train_test = pd.merge(train_test, card, how= 'left', on = 'ID')
train_test = pd.merge(train_test, train_test_score, how = 'left', on = 'ID')
#train_test = pd.merge(train_test, train_test_w2v_feature, how = 'left', on = 'ID')

train = train_test[train_test['MONEY'].notnull()].fillna(-1)
test = train_test[train_test['MONEY'].isnull()].fillna(-1)

xgbModel(config, xgb_param, train, test)
#gbdtModel(config, xgb_param, train, test)
