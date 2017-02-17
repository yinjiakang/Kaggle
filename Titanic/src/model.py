from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

def lrModel(trainX, trainY, testX, data_train):

    logreg = LogisticRegression()

    logreg.fit(trainX, trainY)

    pred = logreg.predict(testX)

    logreg.score(trainX, trainY)

    coeff_df = DataFrame(data_train.columns.delete(0))
    coeff_df.columns = ['Features']
    coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

    return pred

def svmModel(trainX, trainY, testX):

    svc = SVC()

    svc.fit(trainX, trainY)

    pred = svc.predict(testX)

    svc.score(trainX, trainY)

    return pred

def rfModel(trainX, trainY, testX):

    random_forest = RandomForestClassifier(n_estimators=100)

    random_forest.fit(trainX, trainY)

    pred = random_forest.predict(testX)

    random_forest.score(trainX, trainY)

    return pred

def knnModel(trainX, trainY, testX):

    knn = KNeighborsClassifier(n_neighbors = 3)

    knn.fit(trainX, trainY)

    pred = knn.predict(testX)

    knn.score(trainX, trainY)

    return pred

def gauModel(trainX, trainY, testX):

    gaussian = GaussianNB()

    gaussian.fit(trainX, trainY)

    pred = gaussian.predict(testX)

    gaussian.score(trainX, trainY)

    return pred

def xgbModel(trainX, trainY, testX):

    