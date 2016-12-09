from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb 
from sklearn.externals import joblib

random_seed = 1225

def rfModel(features, label, test):

    rf_model = RandomForestRegressor(n_estimators = 100, oob_score = True, min_samples_split = 1, random_state = random_seed)

    rf_model.fit(features, label)

    joblib.dump(rf_model, "../model/rf.model")

    predict = rf_model.predict(test)

    return predict

def gbtModel(features, label, test):

    gbt_model = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.02, max_depth = 7, random_state = random_seed)

    gbt_model.fit(features, label)

    joblib.dump(gbt_model, "../model/gbt.model")

    predict = gbt_model.predict(test)

    return predict

def xgbModel(dtrain, dval, dtest):


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
        'round' : 10
    }


    watchlist = [(dval, "val"), (dtrain, "train")]
    model = xgb.train(params, dtrain, num_boost_round = config['round'], early_stopping_rounds = 100, evals = watchlist)
    model.save_model('../model/xgb_1.model')
    """
    model = xgb.Booster()
    model.load_model("../model/xgb.model")
    """
    test_y = model.predict(dtest, ntree_limit = model.best_ntree_limit)

    return model, test_y