from os import linesep as LS

from name_transformer import NameTransformer

import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def prepare_features(*dataframes):
    prepared = []
    for df in dataframes:
        df = df.copy()

        # Pclass is a categorical feature
        df.Pclass = df.Pclass.astype('category')

        obsolete_columns = ['PassengerId', 'Ticket', 'Cabin']
        if 'Survived' in df.columns:
            # drop the target label from the training set
            obsolete_columns.append('Survived')
        else:
            # ffill that single test sample at index 152 with the missing Fare value
            df.ffill(inplace=True)

        df.drop(columns=obsolete_columns, inplace=True)

        # try to fix the feature skewness as much as possible to achieve a gaussian distribution
        print('feature skewness before transformation:', LS, df.skew(numeric_only=True))
        df.Age = df.Age ** (1/1.27)
        df.Fare = np.log(df.Fare + .45)
        print('feature skewness after transformation:', LS, df.skew(numeric_only=True))

        prepared.append(df)

    return prepared


def create_pipeline():
    column_transformers = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder()), ['Embarked']),
        (make_pipeline(NameTransformer(), OneHotEncoder()), ['Name']),
        (make_pipeline(KNNImputer(), StandardScaler()), ['Age']),
        (make_pipeline(OneHotEncoder()), ['Pclass', 'Sex']),
        (make_pipeline(StandardScaler()), ['Fare', 'Parch', 'SibSp']))

    return column_transformers


def get_models():
    models = {
        'ExtraTreesClassifier': (ExtraTreesClassifier(), {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'n_estimators': [2000]
        }),
        'GaussianNB': (GaussianNB(), {
            'var_smoothing': [1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
        }),
        'LogisticRegression':   (LogisticRegression(), {
            'C': [.01, .1, 1, 10, 100],
            'max_iter': [4000]
        }),
        'SVC': (SVC(), {
            'C': [.01, .1, 1, 10, 100],
            'gamma': [.0001, .001, .01, .1, 1],
            'kernel': ['linear', 'rbf', 'sigmoid']
        })
    }

    return models


def evaluate_models_using_nested_cross_validation(model, param_grid, X, y, scores_dict):
    # sequential execution by the inner loop
    clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(n_splits=10, shuffle=True), error_score='raise')

    # parallel execution by the outer loop
    scores = cross_val_score(clf, X=X, y=y, cv=StratifiedKFold(n_splits=10, shuffle=True), n_jobs=-1, error_score='raise')

    model_name = type(model).__name__
    scores_mean = scores.mean()
    scores_std = scores.std()
    scores_dict[scores_mean] = model
    print(f'accuracy of the model {model_name}: {scores_mean:.3f} (+/- {scores_std:.3f})')


def find_model_hyperparameters_using_cross_validation(model, param_grid, X, y):
    # parallel execution
    grid = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=10, shuffle=True), n_jobs=-1, error_score='raise')
    grid.fit(X, y)
    print(f'best grid searched estimator is: {grid.best_estimator_} with a mean cross-validated score of {grid.best_score_:.3f} and params {grid.best_params_}')

    return grid.best_estimator_


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

X, X_test = prepare_features(train_data, test_data)

pipeline = create_pipeline()
X = pipeline.fit_transform(X)
y = train_data.Survived.to_numpy()

scores_mean = {}
models = get_models()
for model, param_grid in models.values():
    evaluate_models_using_nested_cross_validation(model, param_grid, X, y, scores_mean)

best_score = max(scores_mean)
best_model = scores_mean[best_score]
best_model_name = type(best_model).__name__
print(f'best model score is through: {best_model_name} with a mean accuracy of: {best_score:.3f}')

model, param_grid = models.get(best_model_name)
final_model = find_model_hyperparameters_using_cross_validation(model, param_grid, X, y)

X_test = pipeline.transform(X_test)
y_test = final_model.predict(X_test)

passengerId = test_data.PassengerId
survived = pd.Series(data=y_test)
answer = pd.DataFrame({'PassengerId': passengerId, 'Survived': survived})
answer.to_csv('data/submission.csv', index=False)
print('dumped submission.csv into the current folder')
