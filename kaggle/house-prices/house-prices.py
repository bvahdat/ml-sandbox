from os import linesep as LS

import numpy as np
import pandas as pd

from skewed_transformer import SkewedTransformer

from sklearn import set_config
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from catboost import CatBoostRegressor
from sklearn.linear_model import BayesianRidge, Ridge, OrthogonalMatchingPursuit
from lightgbm import LGBMRegressor


def prepare_features(*dataframes):
    prepared = []
    for df in dataframes:
        df = df.copy()

        # MSSubClass is a categorical feature
        df['MSSubClass'] = df['MSSubClass'].astype('category')

        if 'SalePrice' in df.columns:
            # drop the `Id' & target label from the training set
            df.drop(columns=['Id', 'SalePrice'], inplace=True)

        # impute using 'None' or 'mode' depending on if None has a semantic by the given categorical feature or not
        for column in ['Alley',
                       'BsmtCond',
                       'BsmtExposure',
                       'BsmtFinType1',
                       'BsmtFinType2',
                       'BsmtQual',
                       'Fence',
                       'FireplaceQu',
                       'GarageCond',
                       'GarageFinish',
                       'GarageQual',
                       'GarageType',
                       'MiscFeature',
                       'PoolQC']:
            df[column] = df[column].fillna('None')

        for column in ['Electrical',
                       'Exterior1st',
                       'Exterior2nd',
                       'Functional',
                       'KitchenQual',
                       'MSZoning',
                       'MasVnrType',
                       'SaleType',
                       'Utilities']:
            df[column] = df[column].fillna(df[column].mode()[0])

        # feature engineering
        df['SqFtPerRoom'] = df['GrLivArea'] / \
            (df['TotRmsAbvGrd'] + df['FullBath'] + df['HalfBath'] + df['KitchenAbvGr'])
        df['Total_Home_Quality'] = df['OverallQual'] + df['OverallCond']
        df['Total_Bathrooms'] = (df['FullBath'] + (.5 * df['HalfBath']) +
                                 df['BsmtFullBath'] + (.5 * df['BsmtHalfBath']))
        df['HighQualSF'] = df['1stFlrSF'] + df['2ndFlrSF']

        # 'MoSold' is a cyclic feature
        df['MoSold'] = -np.cos(0.5236 * df['MoSold'])

        prepared.append(df)

    return prepared


def create_pipeline():
    # we expect DataFrame for SkewedTransformer
    set_config(transform_output='pandas')

    return make_column_transformer((make_pipeline(KNNImputer(), SkewedTransformer(), StandardScaler()), make_column_selector(dtype_include=np.number)),
                                   (make_pipeline(OneHotEncoder(sparse_output=False)), make_column_selector(dtype_include='category')))


def get_models():
    catboost_params = {
        'iterations': 6000,
        'learning_rate': 0.005,
        'depth': 4,
        'l2_leaf_reg': 1,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 200,
        'random_seed': 42}

    br_params = {
        'n_iter': 304,
        'tol': 0.16864712769300896,
        'alpha_1': 5.589616542154059e-07,
        'alpha_2': 9.799343618469923,
        'lambda_1': 1.7735725582463822,
        'lambda_2': 3.616928181181732e-06
    }

    lightgbm_params = {
        'num_leaves': 39,
        'max_depth': 2,
        'learning_rate': 0.13705339989856127,
        'n_estimators': 273
    }

    ridge_params = {
        'alpha': 631.1412445239156
    }

    models = {
        'catboost': CatBoostRegressor(**catboost_params, verbose=0),
        'br': BayesianRidge(**br_params),
        'lightgbm': LGBMRegressor(**lightgbm_params),
        'ridge': Ridge(**ridge_params),
        'omp': OrthogonalMatchingPursuit()
    }

    return models


def train(models, X, y):
    for name, model in models.items():
        model.fit(X, y)
        print(name, 'trained')


def evaluate_models(models, X, y):
    results = {}
    for name, model in models.items():
        result = np.exp(np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=10, shuffle=True))))
        results[name] = result

    for name, result in results.items():
        print('-' * 10, name, LS)
        print(np.mean(result))
        print(np.std(result))


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

X, X_test = prepare_features(train_data, test_data)

pipeline = create_pipeline()
X = pipeline.fit_transform(X).to_numpy()
y = train_data.SalePrice.to_numpy()

# target log transformation
y = np.log(y)

models = get_models()
train(models, X, y)
evaluate_models(models, X, y)

X_test = pipeline.transform(X_test)

final_predictions = (
    0.4 * np.exp(models['catboost'].predict(X_test)) +
    0.2 * np.exp(models['br'].predict(X_test)) +
    0.2 * np.exp(models['lightgbm'].predict(X_test)) +
    0.1 * np.exp(models['ridge'].predict(X_test)) +
    0.1 * np.exp(models['omp'].predict(X_test))
)

submission = pd.concat([X_test.Id, pd.Series(final_predictions, name='SalePrice')], axis=1)
submission.to_csv('data/submission.csv', index=False)
print('dumped submission.csv into the current folder')
