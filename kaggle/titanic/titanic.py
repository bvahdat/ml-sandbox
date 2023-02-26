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

train_data = pd.read_csv('http://bit.ly/kaggletrain')

X = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
y = train_data.Survived


def create_pipeline(model):
    column_transformers = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()), ['Age']),
        (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder()), ['Embarked']),
        (make_pipeline(OneHotEncoder()), ['Sex']),
        (make_pipeline(StandardScaler()), ['Fare']), # don't scale 'SibSp', 'Parch', 'Pclass' as it would reduce the accuracy on the test set!
        remainder='passthrough')
    return make_pipeline(column_transformers, model)


def train(X, y, model, scores_dict=None):
    pipeline = create_pipeline(model)
    cv = RepeatedStratifiedKFold(n_splits=10, random_state=1)
    model_name = type(model).__name__
    if scores_dict != None:
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        score_mean = scores.mean()
        score_std = scores.std()
        scores_dict[model_name] = score_mean
        print(f'accuracy of the model {model_name}: {score_mean:.3f} (+/- {score_std:.3f})')
    else:
        pipeline.fit(X, y)
        score = pipeline.score(X, y)
        print(f'score of the model {model_name} with grid searched parameters is {score:.3f}')
        return pipeline


scores_mean = {}
for model in [
    SVC(),
    NuSVC(),
    LinearSVC(max_iter=4000),
    LogisticRegression(),
    GaussianNB(),
    BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
    RandomForestClassifier(n_estimators=10),
    ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0),
    SGDClassifier(loss='hinge', penalty='l2', max_iter=100)
]:
    train(X, y, model, scores_mean)

best_model = max(scores_mean)
print(f'best model score is through {best_model} with a mean of {scores_mean[best_model]:.3f}')

# SVC seems to achieve the best accuracy
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'sigmoid']}
grid = GridSearchCV(SVC(), param_grid, refit=True)
pipeline = train(X, y, grid)

print(f'best SVC grid searched estimator is: {grid.best_estimator_} with a mean cross-validated score of {grid.best_score_:.3f} and params {grid.best_params_}')

y_pred = pipeline.predict(X)
print('confusion matrix:', LS, confusion_matrix(y, y_pred))
print('classification report:', LS, classification_report(y, y_pred))

test_data = pd.read_csv('http://bit.ly/kaggletest')

X_test = test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
X_test.ffill(inplace=True)
y_test = pipeline.predict(X_test)

passengerId = test_data.PassengerId
survived = pd.Series(data=y_test)
answer = pd.DataFrame({'PassengerId': passengerId, 'Survived': survived})
answer.to_csv('./submission.csv', index=False)
print('Dumped submission.csv into the current folder')
