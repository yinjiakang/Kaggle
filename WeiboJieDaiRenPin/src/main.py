import pandas as pd
import numpy as np 
import xgboost as xgb 
from sklearn.cross_validation import train_test_split
from model import *
from function import *


dtrain, dval, dtest, feature_info, featuresType, test_uid, train, label, test= getData()
#print("------finish getdata------")
#model, result = xgbModel(dtrain, dval, dtest)
#print("------finish xgbModel------")
#getFeatureImportance(model, feature_info, featuresType)
#print("------finish getFeatureImportance------")


#result = rfModel(train, label, test)


result = gbtModel(train, label, test)

print("------finish rfModel------")


test_result = pd.DataFrame(columns = ['uid', 'score'])
test_result.uid = test_uid
test_result.score = result

test_result.to_csv("../result/gbt_ori.csv", index = None, encoding = "utf-8")