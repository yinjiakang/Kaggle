import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt  
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

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

    #cv
    #print ('run cv: ' + 'round: ' + str(config['round']) + ' folds: ' + str(config['fold']))
    #res = xgb.cv(params, dtrain, config['round'], nfold = config['fold'], verbose_eval = 100)

    #train
    print ('training: ')
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

    result.to_csv("../data/output/xgb_baseline0213.csv",index=False)


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

    # Add ID
    #drop_columns = ['MONEY']

    
    drop_columns = ['ID', 'MONEY']
    
    """    
    #无投票
    train_features = train.drop(drop_columns, axis = 1).values
    test_features = test.drop(drop_columns, axis = 1).values
    train_label = train['MONEY'].values
    
    clf = GradientBoostingClassifier(n_estimators = 200, max_depth = 6)
    clf.fit(train_features, train_label)
    res = clf.predict(test_features)

    feature_importance
    
    coeff = pd.DataFrame({"columns":list(train.drop(drop_columns, axis = 1).columns), "coef":list(clf.feature_importances_.T)})
    print (coeff)
    """
    
    
    #投票
    train_features = train.drop(drop_columns, axis = 1)
    test_features = test.drop(drop_columns, axis = 1)
    
    # random_features
    top7_columns = [11,6,1,9,0,7,12]
    top8_15_columns = [4,5,7,3,13,8,2,14]
    top7_random_column = random.sample(top7_columns, 4)
    top_8_15_random_column = random.sample(top8_15_columns, 4)
    random_column = top7_random_column + top_8_15_random_column
    # random training data
    random_row = random.sample(range(10885), 5000)
    
    train_features = train_features.iloc[random_row, random_column].values
    test_features = test_features.iloc[:, random_column].values
    train_label = train['MONEY'].values[random_row]
    
    clf = GradientBoostingClassifier(n_estimators = 200, max_depth = 6)
    clf.fit(train_features, train_label)
    res = clf.predict(test_features)
    
    feature_ipt_data = train.drop(drop_columns, axis = 1).iloc[random_row, random_column]
    #feature_importance
    coeff = pd.DataFrame({"columns":list(feature_ipt_data.columns), "coef":list(clf.feature_importances_.T)})
    print (coeff)
    
    
    
    cv_score = cross_validation.cross_val_score(clf, train_features, train_label, cv = 5)
    print ("cv_score:",cv_score)

    """
    #cross_validation
    split_train, split_cv = cross_validation.train_test_split(train, test_size=0.3, random_state=0)

    split_cv_id = split_cv.ID

    split_train_features = split_train.drop(drop_columns, axis = 1).values
    split_train_cv = split_cv.drop(drop_columns, axis = 1).values

    split_train_label = split_train.MONEY
    split_cv_label = split_cv.MONEY


    #bad case
    
    clf.fit(split_train_features, split_train_label)
    cv_pred = clf.predict(split_train_cv)

    bad_result = pd.DataFrame(columns = ["ID","SUBSIDY"])
    bad_result.ID = split_cv_id
    bad_result.SUBSIDY = cv_pred
    bad_result.SUBSIDY = bad_result.SUBSIDY.apply(lambda x:int(x))

    bad_cases = split_cv.loc[split_cv['ID'].drop_duplicates().isin(split_cv[cv_pred != split_cv_label]['ID'].drop_duplicates())]
    bad_cases = pd.merge(bad_cases, bad_result, how='left', on ='ID').drop_duplicates('ID')
    bad_cases_des = bad_cases.loc[:, ['ID', 'MONEY', 'SUBSIDY','CARD_SPEND_SUM', 'CARD_SPEND_MEAN', 'CARD_REMAINDER_SUM', 'CARD_REMAINDER_MEAN','RANK','SCORE' ]]

    bad_cases_des[bad_cases_des['SUBSIDY'] == 0]
    
    #bad_cases_des.to_csv('../data/analysis/bad_cases.csv', index = False)
    """
    

    result = pd.DataFrame(columns = ["studentid","subsidy"])
    result.studentid = test_id
    result.subsidy = res
    result.subsidy = result.subsidy.apply(lambda x:int(x))

    print ('1000--'+str(len(result[result.subsidy==1000])) + ':741')
    print ('1500--'+str(len(result[result.subsidy==1500])) + ':465')
    print ('2000--'+str(len(result[result.subsidy==2000])) + ':354')

    return result
    #result.to_csv("../data/output/gbdt_addw2v0213.csv", index=False)


def rfmodel(config, params, train, test):

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

    # Add ID
    #drop_columns = ['MONEY']

    drop_columns = ['ID', 'MONEY']
    train_features = train.drop(drop_columns, axis = 1).values
    test_features = test.drop(drop_columns, axis = 1).values

    train_label = train['MONEY'].values

    clf = RandomForestClassifier(n_estimators = 200, max_depth = 6)
    clf.fit(train_features, train_label)
    res = clf.predict(test_features)

    #feature_importance
    coeff = pd.DataFrame({"columns":list(train.drop(drop_columns, axis = 1).columns), "coef":list(clf.feature_importances_.T)})
    print (coeff)

    result = pd.DataFrame(columns = ["studentid","subsidy"])
    result.studentid = test_id
    result.subsidy = res
    result.subsidy = result.subsidy.apply(lambda x:int(x))

    print ('1000--'+str(len(result[result.subsidy==1000])) + ':741')
    print ('1500--'+str(len(result[result.subsidy==1500])) + ':465')
    print ('2000--'+str(len(result[result.subsidy==2000])) + ':354')

    return result
