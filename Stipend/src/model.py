import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt  
from sklearn import cross_validation

def xgbModel(config, params, train, test):

    train_id = train.ID
    test_id = test.ID

    drop_columns = ['ID', 'MONEY']
    train_features = train.drop(drop_columns, axis = 1)
    test_features = test.drop(drop_columns, axis = 1)

    train_label = train.MONEY
    train_id = train.ID
    test_id = test.ID

    #encoding label
    le = preprocessing.LabelEncoder()
    train_encode_label = le.fit_transform(train_label)

    dtrain = xgb.DMatrix(train_features, label = train_encode_label)
    dtest = xgb.DMatrix(test_features)

    #for balance
    ssy0 = train[train['MONEY'] == 0]['ID'].count()
    ssy1000 = train[train['MONEY'] == 1000]['ID'].count()
    ssy1500 = train[train['MONEY'] == 1500]['ID'].count()
    ssy2000 = train[train['MONEY'] == 2000]['ID'].count()
    ssyNum = train['ID'].count()

    #cv
    #print ('run cv: ' + 'round: ' + str(config['round']) + ' folds: ' + str(config['fold']))
    #res = xgb.cv(params, dtrain, config['round'], nfold = config['fold'], verbose_eval = 100)

    #train
    watchlist = [ (dtrain,'train')]
    xgbmodel = xgb.train(params, dtrain, config['round'], watchlist, verbose_eval = 100)
    pred = xgbmodel.predict(dtest)
    intpred = [int(pred[i]) for i in range(len(pred))]
    real_pred = le.inverse_transform(intpred)

    print( "--- plot_importance ---")

    xgb.plot_importance(xgbmodel)
    plt.show()

    print( "--- plot_importance finish ---")


    result = pd.DataFrame(columns = ["studentid","subsidy"])
    result.studentid = test_id
    result.subsidy = real_pred
    result.subsidy = result.subsidy.apply(lambda x:int(x))

    print ('1000--'+str(len(result[result.subsidy==1000])) + ':741')
    print ('1500--'+str(len(result[result.subsidy==1500])) + ':465')
    print ('2000--'+str(len(result[result.subsidy==2000])) + ':354')

    result.to_csv("../data/output/xgb_baseline.csv",index=False)

def gbdtModel(config, params, train, test):

    Oversampling1000 = train.loc[train.MONEY == 1000]
    Oversampling1500 = train.loc[train.MONEY == 1500]
    Oversampling2000 = train.loc[train.MONEY == 2000]
    for i in range(5):
        train = train.append(Oversampling1000)
    for j in range(8):
        train = train.append(Oversampling1500)
    for k in range(10):
        train = train.append(Oversampling2000)

    test_id = test.ID

    drop_columns = ['ID', 'MONEY']
    train_features = train.drop(drop_columns, axis = 1).values
    test_features = test.drop(drop_columns, axis = 1).values

    train_label = train['MONEY'].values

    clf = GradientBoostingClassifier(n_estimators = 200, max_depth = 6)
    clf.fit(train_features, train_label)
    res = clf.predict(test_features)

    #feature_importance
    coeff = pd.DataFrame({"columns":list(train.drop(drop_columns, axis = 1).columns), "coef":list(clf.feature_importances_.T)})
    print (coeff)

    split_train, split_cv = cross_validation.train_test_split(train, test_size=0.3, random_state=0)
    split_train_features = split_train.drop(drop_columns, axis = 1).values
    split_train_cv = split_cv.drop(drop_columns, axis = 1).values

    split_train_label = split_train.MONEY
    split_cv_label = split_cv.MONEY

    cv_score = cross_validation.cross_val_score(clf, train_features, train_label, cv = 5)

    print (cv_score)
    #bad case

"""    result = pd.DataFrame(columns = ["studentid","subsidy"])
    result.studentid = test_id
    result.subsidy = res
    result.subsidy = result.subsidy.apply(lambda x:int(x))

    result.to_csv("../data/output/gbdt_baseline_r200.csv", index=False)"""