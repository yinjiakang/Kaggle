import pandas as pd
import numpy as np
import xgboost as  xgb
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
%matplotlib inline

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

data_train = pd.read_csv('../data/input/train.csv')
data_test = pd.read_csv('../data/input/test.csv')
droplist_titanic = []
droplist_test = []

"""
# notebook
data_train
data_train.info()
data_train.describe()
"""

#Embarked
sns.factorplot('Embarked', 'Survived', data = data_train, size = 4, aspect = 3)

fig, (axis1, axis2, axis3) = plt.subplots(1,3, figsize=(15,5))

sns.countplot('Embarked', data = data_train, ax = axis1)
sns.countplot('Survived', hue = 'Embarked', data = data_train, ax = axis2)

embark_sur = data_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()
#embark_sur
sns.barplot(x = 'Embarked', y = 'Survived', data = embark_sur, ax = axis3, order = ['S','C','Q'])

embark_dummies_titanic = pd.get_dummies(data_train['Embarked'])
#embark_dummies_titanic
embark_dummies_titanic.drop(['S'], axis = 1, inplace=True)

embark_dummies_test = pd.get_dummies(data_test['Embarked'])
#embark_dummies_test
embark_dummies_test.drop(['S'],axis = 1, inplace=True)

data_train = data_train.join(embark_dummies_titanic)
data_test = data_test.join(embark_dummies_test)

droplist_titanic.append('Embarked')
droplist_test.append('Embarked')

#embark_dummies_titanic
#data_test.info()

data_test['Fare'].fillna(data_test['Fare'].median(), inplace = True)

data_train['Fare'] = data_train['Fare'].astype(int)
data_test['Fare'] = data_test['Fare'].astype(int)

fare_not_survived = data_train['Fare'][data_train['Survived'] == 0]
fare_survived = data_train['Fare'][data_train['Survived'] == 1]

average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])

#plot
data_train['Fare'].plot(kind = 'hist', figsize=(15,3),bins=100, xlim=(0,100))

average_fare.index.names = std_fare.index.names = ["Survived"]
average_fare.plot(yerr=std_fare,kind='bar',legend=False)

#Age
fig, (axis1, axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age Values - Titanic')
axis2.set_title('New Age Values - Titanic')

# average , std, count nan in train 
average_age_titanic = data_train['Age'].mean()
std_age_titanic = data_train['Age'].std()
count_nan_age_titanic = data_train['Age'].isnull().sum()

# average , std, count nan in test  
average_age_test = data_test['Age'].mean()
std_age_test = data_test['Age'].std()
count_nan_age_test = data_test['Age'].isnull().sum()

rand_age_titanic = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_age_test = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

data_train['Age'].dropna().astype(int).hist(bins = 70, ax = axis1)

data_train['Age'][np.isnan(data_train['Age'])]  = rand_age_titanic
data_test['Age'][np.isnan(data_test['Age'])] = rand_age_test

data_train['Age'] = data_train['Age'].astype(int)
data_test['Age'] = data_test['Age'].astype(int)

# continue with age 
facet = sns.FacetGrid(data_train, hue = 'Survived', aspect = 4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim = (0, data_train['Age'].max()))
facet.add_legend()

fig ,axis = plt.subplots(1,1,figsize=(18,4))
age_average_survived_titanic = data_train[['Age', 'Survived']].groupby(['Age'], as_index = False).mean()
sns.barplot(x = 'Age', y = 'Survived', data = age_average_survived_titanic)
data_train['Age'].hist(bins = 70, ax = axis2)

#Family
data_train['Family'] = data_train['SibSp'] + data_train['Parch']
data_train['Family'].loc[data_train['Family'] > 0] = 1
data_train['Family'].loc[data_train['Family'] == 0] = 0

data_test['Family'] =  data_test["Parch"] + data_test["SibSp"]
data_test['Family'].loc[data_test['Family'] > 0] = 1
data_test['Family'].loc[data_test['Family'] == 0] = 0


droplist_titanic += ['Cabin', 'SibSp', 'Parch']
droplist_test += ['Cabin', 'SibSp', 'Parch']

fig, (axis1, axis2) = plt.subplots(1,2,sharex = True, figsize=(10,5))
axis1.set_title('有无亲人')
axis2.set_title('获救概率')

sns.countplot(x = 'Family', data=data_train, order = [1, 0], ax=axis1)

family_prec = data_train[['Family', 'Survived']].groupby(['Family'], as_index = False).mean()
sns.barplot(x = 'Family', y = 'Survived', data=family_prec, order = [1, 0], ax = axis2)

axis1.set_xticklabels(['With Family', 'Alone'], rotation=0)

#Sex
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

data_train['Person'] = data_train[['Age', 'Sex']].apply(get_person, axis = 1)
data_test['Person'] = data_test[['Age', 'Sex']].apply(get_person, axis = 1)

droplist_titanic.append('Sex')
droplist_test.append('Sex')


dummies_person_titanic = pd.get_dummies(data_train['Person'])
dummies_person_titanic.columns = ['Child', 'Female', 'Male']
dummies_person_titanic.drop(['Male'], axis = 1, inplace=True)


dummies_person_test = pd.get_dummies(data_test['Person'])
dummies_person_test.columns = ['Child', 'Female', 'Male']
dummies_person_test.drop(['Male'], axis = 1, inplace=True)

data_train = data_train.join(dummies_person_titanic)
data_test = data_test.join(dummies_person_test)

fig, (axis1, axis2) = plt.subplots(1, 2, figsize= (12,5))

sns.countplot(x = 'Person', data = data_train, ax = axis1)

person_prec = data_train[['Person', 'Survived']].groupby(['Person'], as_index = False).mean()

sns.barplot(x = 'Person', y = 'Survived', data=person_prec, ax=axis2)

droplist_titanic.append('Person')
droplist_test.append('Person')


#Pclass
sns.factorplot('Pclass', 'Survived', order = [1,2,3], data= data_train, size= 5)

pclass_dummies_titanic = pd.get_dummies(data_train['Pclass'])
pclass_dummies_titanic.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis = 1,inplace=True)

pclass_dummies_test = pd.get_dummies(data_test['Pclass'])
pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_test.drop(['Class_3'], axis = 1,inplace=True)

#Name
title_titanic = data_train['Name'].str.extract(', (.*)\.')
title_test = data_test['Name'].str.extract(', (.*)\.')

data_train['Title'] = title_titanic
data_test['Title'] = title_test



droplist_titanic.append('Pclass')
droplist_test.append('Pclass')

data_train = data_train.join(pclass_dummies_titanic)
data_test = data_test.join(pclass_dummies_test)

droplist_titanic += ['PassengerId', 'Survived', 'Name', 'Ticket']
droplist_test += ['PassengerId', 'Name', 'Ticket']

#train
train_label = data_train['Survived']
train_features = data_train.drop(droplist_titanic, axis = 1)

#test
testID = data_test['PassengerId']
test_features = data_test.drop(droplist_test, axis = 1)

logreg = LogisticRegression()
logreg.fit(train_features, train_label)
pred = logreg.predict(test_features)
logreg.score(train_features, train_label)



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train_features, train_label)

Y_pred = random_forest.predict(test_features)

random_forest.score(train_features, train_label)

coeff_df = pd.DataFrame(train_features.columns)
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

print ("coeff_df: ", coeff_df)

result = pd.DataFrame(columns = ['PassengerId', 'Survived'])
result.PassengerId = testID
result.Survived = Y_pred

result.to_csv('../data/output/RF_baseline.csv', index = False)