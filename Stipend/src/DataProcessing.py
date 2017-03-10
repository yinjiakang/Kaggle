import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from collections import Counter
import datetime


def translate_time(full_date):
    time = full_date.split(' ')
    base_time = datetime.datetime.strptime('00:00:00', "%X")
    cur_time = datetime.datetime.strptime(time[1] , "%X")
    sec = (cur_time - base_time).total_seconds()
    return sec

def judge_dorm(line):
    time = line[1]
    way = line[2]
    state = -1
    if way == 0:
        if line[1] <= 21600.0 or line[1] >= 82600.0:
            state = 0
        else:
            state = 1
    else:
        if line[1] <= 18000.0 or line[1] >= 79200.0:
            state = 0
        else:
            state = 1

    return state

#read subsidy
train_subsidy = pd.read_csv('../data/train/subsidy_train.txt', header = None)
train_subsidy.columns = ['ID', 'MONEY']
test_subsidy = pd.read_csv('../data/test/studentID_test.txt', header = None)
test_subsidy.columns = ['ID']
test_subsidy['MONEY'] = np.nan
train_test = pd.concat([train_subsidy, test_subsidy])

train_test.to_csv("../data/input/train_test.csv", index = False)

#read card
train_card = pd.read_csv('../data/train/card_train.txt', header = None)
train_card.columns = ['ID', 'CARD_CAT', 'CARD_WHERE', 'CARD_HOW', 'CARD_TIME', 'CARD_SPEND', 'CARD_REMAINDER']
test_card = pd.read_csv('../data/test/card_test.txt', header = None)
test_card.columns = ['ID', 'CARD_CAT', 'CARD_WHERE', 'CARD_HOW', 'CARD_TIME', 'CARD_SPEND', 'CARD_REMAINDER']
card_train_test = pd.concat([train_card,test_card])


#process card data
card = pd.DataFrame(card_train_test.groupby(['ID'])['CARD_CAT'].count())

card['CARD_SPEND_SUM'] = card_train_test.groupby(['ID'])['CARD_SPEND'].sum()
card['CARD_SPEND_MEAN'] = card_train_test.groupby(['ID'])['CARD_SPEND'].mean()
card['CARD_SPEND_STD'] = card_train_test.groupby(['ID'])['CARD_SPEND'].max()
card['CARD_SPEND_MEDIAN'] = card_train_test.groupby(['ID'])['CARD_SPEND'].median()

card['CARD_REMAINDER_SUM'] = card_train_test.groupby(['ID'])['CARD_REMAINDER'].sum()
card['CARD_REMAINDER_MEAN'] = card_train_test.groupby(['ID'])['CARD_REMAINDER'].mean()
card['CARD_REMAINDER_STD'] = card_train_test.groupby(['ID'])['CARD_REMAINDER'].max()
card['CARD_REMAINDER_MEDIAN'] = card_train_test.groupby(['ID'])['CARD_REMAINDER'].median()

card.to_csv('../data/input/cardInfo.csv', index = True)
card = pd.read_csv('../data/input/cardInfo.csv')


#read score
train_score = pd.read_csv('../data/train/score_train.txt', header = None)
train_score.columns = ['ID', 'COLLEGE', 'RANK']
test_score = pd.read_csv('../data/test/score_test.txt', header = None)
test_score.columns = ['ID', 'COLLEGE', 'RANK']
train_test_score = pd.concat([train_score, test_score])

score = pd.DataFrame(train_test_score.groupby(['COLLEGE'])['RANK'].max())
score.to_csv('../data/input/collegeInfo.csv', index = True)
score = pd.read_csv('../data/input/collegeInfo.csv')
score.columns = ['COLLEGE', 'COLLEGE_STU_NUM']

train_test_score = pd.merge(train_test_score, score, how='left', on='COLLEGE')
train_test_score['SCORE'] = train_test_score['RANK'] / train_test_score['COLLEGE_STU_NUM']
train_test_score.to_csv("../data/input/train_test_score.csv", index = False)

#read library
train_library = pd.read_csv('../data/train/library_train.txt', header = None)
train_library.columns = ['ID', 'DOOR', 'DOOR_TIME']
test_library = pd.read_csv('../data/test/library_test.txt', header = None)
test_library.columns = ['ID', 'DOOR', 'DOOR_TIME']
train_test_library = pd.concat([train_library, test_library])

go_library_num = train_test_library['ID'].groupby(train_test_library['ID']).count()
go_library_num = pd.DataFrame(go_library_num)
go_library_num.columns = ['LIB_NUM']

go_library_num.to_csv('../data/input/library.csv', index = True)


#read dorm
train_dorm = pd.read_csv('../data/train/dorm_train.txt', header = None)
train_dorm.columns = ['ID','DORM_TIME', 'IN_OUT']
test_dorm = pd.read_csv('../data/test/dorm_test.txt', header = None)
test_dorm.columns = ['ID','DORM_TIME', 'IN_OUT']
train_test_dorm = pd.concat([train_dorm, test_dorm])

train_test_dorm['DORM_TIME'] = train_test_dorm['DORM_TIME'].apply(translate_time)
train_test_dorm['IF_NORMAL'] = train_test_dorm.apply(judge_dorm, axis = 1)

train_test_dorm.to_csv('../data/analysis/full_train_test_dorm.csv', index = False)

train_test_dorm = train_test_dorm.drop(['DORM_TIME', 'IN_OUT'], axis = 1)
temp = train_test_dorm[train_test_dorm['IF_NORMAL'] == 0]
not_normal = temp.groupby(temp['ID']).count()
not_normal.columns = ['NOT_NORMAL_NUM']

not_normal.to_csv('../data/input/not_normal.csv')