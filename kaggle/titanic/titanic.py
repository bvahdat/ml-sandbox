from os import linesep as LS

import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
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
        copy['Title'] = copy['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Infrequent')
        copy['Title'] = copy['Title'].replace('Mlle', 'Miss')
        copy['Title'] = copy['Title'].replace('Ms', 'Miss')
        copy['Title'] = copy['Title'].replace('Mme', 'Mrs')
        copy.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        prepared.append(copy)
    return prepared


X, X_test = prepare_features(train_data, test_data)

# drop the target label from the training set
X.drop(columns=['Survived'], inplace=True)

# ffill that test sample at index 152 with missing Fare value
X_test.ffill(inplace=True)


def create_pipeline():
    column_transformers = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()), ['Age']),
        (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder()), ['Embarked']),
        (make_pipeline(OneHotEncoder()), ['Sex', 'Title']),
        (make_pipeline(StandardScaler()), ['Fare', 'Parch', 'Pclass', 'SibSp']))
    return make_pipeline(column_transformers)


def evaluate_model_using_nested_cross_validation(model, param_grid, X, y, scores_dict):
    inner_cv = StratifiedKFold(shuffle=True)
    outer_cv = StratifiedKFold(shuffle=True)
    clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)
    scores = cross_val_score(clf, X=X, y=y, cv=outer_cv, n_jobs=-1, error_score='raise')
    model_name = type(model).__name__
    scores_mean = scores.mean()
    scores_std = scores.std()
    scores_dict[scores_mean] = model
    print(f'accuracy of the model {model_name}: {scores_mean:.3f} (+/- {scores_std:.3f})')


pipeline = create_pipeline()
X = pipeline.fit_transform(X)
y = y.to_numpy()

models = [
    ExtraTreesClassifier(),
    GaussianNB(),
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
    XGBClassifier()
]

models_param_grid = [
    {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'n_estimators': [2000]
    },
    {
        'var_smoothing': [1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    },
    {
        'C': [.01, .1, 1, 10, 100],
        'max_iter': [4000]
    },
    {
        'bootstrap': [True, False],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [2000]
    },
    {
        'C': [.01, .1, 1, 10, 100],
        'degree': [2, 3, 4, 5],
        'gamma': [.0001, .001, .01, .1, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    },
    {
        'colsample_bytree': [0.3, 0.5, 0.8],
        'reg_alpha': [0, 0.5, 1, 5],
        'reg_lambda': [0, 0.5, 1, 5],
        'n_estimators': [2000]
    }
]

scores_mean = {}

for model, param_grid in zip(models, models_param_grid):
    evaluate_model_using_nested_cross_validation(model, param_grid, X, y, scores_mean)

final_score = max(scores_mean)
final_model = scores_mean[final_score]
final_model_name = type(final_model).__name__
print(f'best model score is through: {final_model_name} with a mean accuracy of: {final_score:.3f} and parameters: {final_model.get_params()}')

final_model = scores_mean[final_score]
final_model.fit(X, y)

X_test = pipeline.transform(X_test)
y_test = final_model.predict(X_test)

passengerId = test_data.PassengerId
survived = pd.Series(data=y_test)
answer = pd.DataFrame({'PassengerId': passengerId, 'Survived': survived})
answer.to_csv('./submission.csv', index=False)
print('dumped submission.csv into the current folder')
