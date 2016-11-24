import pandas as pd
import numpy as np 
import xgboost as xgb 
from sklearn.cross_validation import train_test_split

random_seed = 1225

trainxFile = "../data/train_x.csv"
trainyFile = "../data/train_y.csv"
testxFile = "../data/test_x.csv"
featuresTypeFile = "../data/features_type.csv"

trainx = pd.read_csv(trainxFile)
trainy = pd.read_csv(trainyFile)
trainxy = pd.merge(trainx, trainy, on = "uid")


testx = pd.read_csv(testxFile)
test_uid = testx.uid
testx = testx.drop(['uid'], axis = 1)


featuresType = pd.read_csv(featuresTypeFile)
featuresType.index = featuresType.feature
featuresType = featuresType.drop(['feature'], axis = 1)
featuresType = featuresType.to_dict()['type']

featureName = list(trainx.columns)
featureName.remove("uid")

feature_info = {}

for feature in featureName:
    max = trainx[feature].max()
    min = trainx[feature].min()
    n_null = len(trainx[trainx[feature] < 0])
    n_gt1w = len(trainx[trainx[feature] > 10000])
    feature_info[feature] = [max, min, n_null, n_gt1w]

print ("neg:{0},pos:{1}".format(len(trainxy[trainxy.y==0]),len(trainxy[trainxy.y==1])))


trainxy = trainxy.drop(['uid'], axis = 1)
train, val = train_test_split(trainxy, test_size = 0.2, random_state = random_seed)

train_label = train.y
x = train.drop(['y'], axis = 1)

val_label = val.y
val_x = val.drop(['y'], axis = 1)



dtest = xgb.DMatrix(testx)
dval = xgb.DMatrix(val_x, label = val_label)
dtrain = xgb.DMatrix(x, label = train_label) 

params = {
    "booster":'gbtree',
    'objective': 'binary:logistic',
    'early_stopping_rounds':100,
    'scale_pos_weight': 1400.0/13458.0,
    'eval_metric': 'auc',

    'gamma':0.1,
    'min_child_weight':3,

    'max_depth':8,
    'lambda':550,
    'subsample':0.7,    
    'colsample_bytree':0.4,
    
    'eta': 0.02,
    'seed': random_seed,
    'nthread': 3
}

config = {
    'round' : 10000
}


watchlist = [(dval, "val"), (dtrain, "train")]
model = xgb.train(params, dtrain, num_boost_round = config['round'], early_stopping_rounds = 100, evals = watchlist)
model.save_model('../model/xgb.model')
"""
model = xgb.Booster()
model.load_model("../model/xgb.model")
"""

test_y = model.predict(dtest, ntree_limit = model.best_ntree_limit)
test_result = pd.DataFrame(columns = ['uid', 'score'])
test_result.uid = test_uid
test_result.score = test_y
test_result.to_csv("../data/xgb_ori.csv", index = None, encoding = "utf-8")

feature_score = model.get_fscore()

print ("------------------------------------------------\n")
print (feature_score)
print ("------------------------------------------------\n")

for key in feature_score:
    feature_score[key] = [feature_score[key]] + feature_info[key] + [featuresType[key]]

print (len(feature_score[key]))

feature_score = sorted(feature_score.items(), key = lambda x: x[1], reverse = True)
fs = []

for (key, value) in feature_score:
    fs.append("{0},{1},{2},{3},{4},{5},{6}\n".format(key,value[0],value[1],value[2],value[3],value[4],value[5]))

with open('../data/feature_score.csv','w') as f:
    f.writelines("feature,score,min,max,n_null,n_gt1w\n")
    f.writelines(fs)
