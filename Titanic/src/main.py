import pandas as pd     

data_train = pd.read_csv('../data/analysis/data_train.csv')
data_test = pd.read_csv('../data/analysis/data_test.csv')

droplist_titanic = ['Embarked', 'Cabin', 'SibSp', 'Parch', 'Sex', 'Person', 'Pclass']
droplist_test = ['Embarked', 'Cabin', 'SibSp', 'Parch', 'Sex', 'Person', 'Pclass']

droplist_titanic += ['PassengerId', 'Survived', 'Name', 'Ticket']
droplist_test += ['PassengerId', 'Name', 'Ticket']

#train
train_label = data_train['Survived']
train_features = data_train.drop(droplist_titanic, axis = 1)

#test
testID = data_test['PassengerId']
test_features = data_test.drop(droplist_test, axis = 1)

#Logistic Regression
