from os import linesep as LS

import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, NuSVC, LinearSVC
from xgboost import XGBClassifier

train_data = pd.read_csv('http://bit.ly/kaggletrain')
test_data = pd.read_csv('http://bit.ly/kaggletest')

y = train_data.Survived


def prepare_features(train_df, test_df):
    prepared = []
    for dataframe in [train_df, test_df]:
        copy = dataframe.copy()
        copy['Title'] = copy.Name.str.extract('([A-Za-z]+)\.', expand=False)
        copy['Title'] = copy['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Infrequent')
        copy['Title'] = copy['Title'].replace('Mlle', 'Miss')
        copy['Title'] = copy['Title'].replace('Ms', 'Miss')
        copy['Title'] = copy['Title'].replace('Mme', 'Mrs')
        copy.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        prepared.append(copy)
    return prepared


X, X_test = prepare_features(train_data, test_data)

# drop the target label from the training set
X.drop(columns=['Survived'], inplace=True)

# drop that one single test sample at index 152 with Fare = NaN
X_test.ffill(inplace=True)


def create_pipeline(model):
    column_transformers = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()), ['Age']),
        (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder()), ['Embarked']),
        (make_pipeline(OneHotEncoder()), ['Sex', 'Title']),
        (make_pipeline(StandardScaler()), ['Fare', 'Parch', 'Pclass', 'SibSp']))
    return make_pipeline(column_transformers, model)


def train(X, y, model, scores_dict=None):
    pipeline = create_pipeline(model)
    cv = RepeatedStratifiedKFold(n_splits=10)
    model_name = type(model).__name__
    if scores_dict != None:
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        score_mean = scores.mean()
        score_std = scores.std()
        scores_dict[score_mean] = model_name
        print(f'accuracy of the model {model_name}: {score_mean:.3f} (+/- {score_std:.3f})')
    else:
        pipeline.fit(X, y)
        score = pipeline.score(X, y)
        print(f'score of the model {model_name} with grid searched parameters is {score:.3f}')
        return pipeline


scores_mean = {}
for model in [
    BaggingClassifier(KNeighborsClassifier(n_neighbors=3)),
    ExtraTreesClassifier(n_estimators=2000),
    GaussianNB(),
    LinearSVC(max_iter=4000),
    LogisticRegression(),
    NuSVC(),
    RandomForestClassifier(n_estimators=2000),
    SGDClassifier(max_iter=4000),
    SVC(),
    XGBClassifier(n_estimators=2000)
]:
    train(X, y, model, scores_mean)

best_score = max(scores_mean)
print(f'best model score is through {scores_mean[best_score]} with a mean of {best_score:.3f}')

# SVC seems to achieve the best accuracy
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'sigmoid']}
grid = GridSearchCV(SVC(), param_grid)
pipeline = train(X, y, grid)

print(f'best SVC grid searched estimator is: {grid.best_estimator_} with a mean cross-validated score of {grid.best_score_:.3f} and params {grid.best_params_}')

y_pred = pipeline.predict(X)
print('confusion matrix:', LS, confusion_matrix(y, y_pred))
print('classification report:', LS, classification_report(y, y_pred))

y_test = pipeline.predict(X_test)

passengerId = test_data.PassengerId
survived = pd.Series(data=y_test)
answer = pd.DataFrame({'PassengerId': passengerId, 'Survived': survived})
answer.to_csv('./submission.csv', index=False)
print('Dumped submission.csv into the current folder')
