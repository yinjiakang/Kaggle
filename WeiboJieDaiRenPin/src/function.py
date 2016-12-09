import pandas as pd
import numpy as np 
import xgboost as xgb 
from sklearn.cross_validation import train_test_split
from model import *


def getData():

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


    #data processing for xgb
    train_uid = trainxy.uid
    trainxy = trainxy.drop(['uid'], axis = 1)
    train, val = train_test_split(trainxy, test_size = 0.2, random_state = random_seed)

    train_label = train.y
    x = train.drop(['y'], axis = 1)

    val_label = val.y
    val_x = val.drop(['y'], axis = 1)

    #date format of xgb
    dtest = xgb.DMatrix(testx)
    dval = xgb.DMatrix(val_x, label = val_label)
    dtrain = xgb.DMatrix(x, label = train_label) 

    #data format of randomforest
    rflabel = trainxy.y
    rftrain = trainxy.drop(['y'], axis = 1)
    rftest = testx


    rflabel.to_csv("../data/alltrainlabel.csv", index = False)
    rftrain.to_csv("../data/alltrainfeature.csv", index = False)
    rftest.to_csv("../data/alltestfeature.csv", index = False)
    test_uid.to_csv("../data/test_uid.csv", index = False)
    train_uid.to_csv("../data/train_uid.csv", index = False)

    return dtrain, dval, dtest, feature_info, featuresType, test_uid, rftrain, rflabel, rftest



def getFeatureImportance(model, feature_info, featuresType):

    feature_score = model.get_fscore()

    for key in feature_score:
        feature_score[key] = [feature_score[key]] + feature_info[key] + [featuresType[key]]

    feature_score = sorted(feature_score.items(), key = lambda x: x[1], reverse = True)
    fs = []

    for (key, value) in feature_score:
        fs.append("{0},{1},{2},{3},{4},{5},{6}\n".format(key,value[0],value[1],value[2],value[3],value[4],value[5]))

    with open('../data/feature_score_xxxx.csv','w') as f:
        f.writelines("feature,score,min,max,n_null,n_gt1w\n")
        f.writelines(fs)
