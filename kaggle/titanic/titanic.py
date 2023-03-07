import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


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


def create_pipeline():
    column_transformers = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()), ['Age']),
        (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder()), ['Embarked']),
        (make_pipeline(OneHotEncoder()), ['Sex', 'Title']),
        (make_pipeline(StandardScaler()), ['Fare', 'Parch', 'Pclass', 'SibSp']))
    return make_pipeline(column_transformers)


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
        'RandomForestClassifier': (RandomForestClassifier(), {
            'bootstrap': [True, False],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [2000]
        }),
        'SVC': (SVC(), {
            'C': [.01, .1, 1, 10, 100],
            'degree': [2, 3, 4, 5],
            'gamma': [.0001, .001, .01, .1, 1],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }),
        'XGBClassifier': (XGBClassifier(), {
            'colsample_bytree': [0.3, 0.5, 0.8],
            'reg_alpha': [0, 0.5, 1, 5],
            'reg_lambda': [0, 0.5, 1, 5],
            'n_estimators': [2000]
        })
    }

    return models


def evaluate_models_using_nested_cross_validation(model, param_grid, X, y, scores_dict):
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True)

    # sequential execution by the inner loop
    clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)

    # parallel execution by the outer loop
    scores = cross_val_score(clf, X=X, y=y, cv=outer_cv, n_jobs=-1, error_score='raise')

    model_name = type(model).__name__
    scores_mean = scores.mean()
    scores_std = scores.std()
    scores_dict[scores_mean] = model
    print(f'accuracy of the model {model_name}: {scores_mean:.3f} (+/- {scores_std:.3f})')


def find_model_hyperparameters_using_cross_validation(model, param_grid, X, y):
    grid = GridSearchCV(model, param_grid)
    grid.fit(X, y)
    print(f'best grid searched estimator is: {grid.best_estimator_} with a mean cross-validated score of {grid.best_score_:.3f} and params {grid.best_params_}')
    return grid.best_estimator_


train_data = pd.read_csv('http://bit.ly/kaggletrain')
test_data = pd.read_csv('http://bit.ly/kaggletest')

y = train_data.Survived

X, X_test = prepare_features(train_data, test_data)

# drop the target label from the training set
X.drop(columns=['Survived'], inplace=True)

# ffill that single test sample at index 152 with missing Fare value
X_test.ffill(inplace=True)

pipeline = create_pipeline()
X = pipeline.fit_transform(X)
y = y.to_numpy()

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
answer.to_csv('./submission.csv', index=False)
print('dumped submission.csv into the current folder')
