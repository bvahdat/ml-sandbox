import numpy as np
import pandas as pd

from scipy.stats import randint, uniform

from skewed_transformer import SkewedTransformer

from sklearn import set_config
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from catboost import CatBoostRegressor
from sklearn.linear_model import BayesianRidge, Ridge, OrthogonalMatchingPursuit
from lightgbm import LGBMRegressor


def prepare_features(dataframe):
    df = dataframe.copy()

    # 'MSSubClass' is a categorical feature
    df['MSSubClass'] = df['MSSubClass'].astype('category')

    # 'MoSold' is a cyclic feature
    df['MoSold'] = -np.cos(.5236 * df['MoSold'])

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
    df['SqFtPerRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + df['FullBath'] + df['HalfBath'] + df['KitchenAbvGr'])
    df['Total_Home_Quality'] = df['OverallQual'] + df['OverallCond']
    df['Total_Bathrooms'] = (df['FullBath'] + (.5 * df['HalfBath']) + df['BsmtFullBath'] + (.5 * df['BsmtHalfBath']))
    df['HighQualSF'] = df['1stFlrSF'] + df['2ndFlrSF']

    return df


def create_pipeline():
    # we expect DataFrame for SkewedTransformer
    set_config(transform_output='pandas')

    return make_column_transformer((make_pipeline(KNNImputer(), SkewedTransformer(), StandardScaler()), make_column_selector(dtype_include=np.number)),
                                   (make_pipeline(OneHotEncoder(sparse_output=False)), make_column_selector(dtype_include='category')))


def create_models(X, y):
    catboost_params_search_space = dict(iterations=randint(5000, 7000),
                                        learning_rate=uniform(loc=.003, scale=.004),
                                        depth=randint(3, 5))

    br_params_search_space = dict(n_iter=randint(250, 350),
                                  tol=uniform(loc=.1, scale=.2),
                                  alpha_1=uniform(loc=3.0, scale=6.0),
                                  alpha_2=uniform(loc=7.0, scale=12.0),
                                  lambda_1=uniform(loc=1.0, scale=2.0),
                                  lambda_2=uniform(loc=3.0, scale=5.0))

    lightgbm_params_search_space = dict(num_leaves=randint(40, 50),
                                        max_depth=randint(1, 3),
                                        learning_rate=uniform(loc=.1, scale=.2),
                                        n_estimators=randint(250, 300))

    ridge_params_search_space = dict(alpha=uniform(loc=600.0, scale=100.0))

    omp_params_search_space = dict(n_nonzero_coefs=randint(10, 20))

    models_search_space = {
        'CatBoostRegressor': (CatBoostRegressor(eval_metric='RMSE', verbose=0), catboost_params_search_space),
        'BayesianRidge': (BayesianRidge(), br_params_search_space),
        'LGBMRegressor': (LGBMRegressor(), lightgbm_params_search_space),
        'Ridge': (Ridge(), ridge_params_search_space),
        'OrthogonalMatchingPursuit': (OrthogonalMatchingPursuit(), omp_params_search_space)
    }

    models = {}
    for model_name, model_params in models_search_space.items():
        clf = RandomizedSearchCV(model_params[0], model_params[1], scoring='neg_mean_squared_error', error_score='raise')
        search = clf.fit(X, y)
        models[model_name] = search.best_estimator_
        print(f'RMSE of the model {model_name}: {-search.best_score_:.3f} using the params: ({search.best_params_})')

    return models


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


X_all = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
X_all = prepare_features(X_all)

pipeline = create_pipeline()
X_all = pipeline.fit_transform(X_all).to_numpy()

train_data_len = len(train_data.index)
X = X_all[:train_data_len, :]
y = train_data.SalePrice.to_numpy()

# target log transformation
y = np.log(y)

models = create_models(X, y)

X_test = X_all[train_data_len:, :]
predictions = (
    .4 * np.exp(models['CatBoostRegressor'].predict(X_test)) +
    .2 * np.exp(models['BayesianRidge'].predict(X_test)) +
    .2 * np.exp(models['LGBMRegressor'].predict(X_test)) +
    .1 * np.exp(models['Ridge'].predict(X_test)) +
    .1 * np.exp(models['OrthogonalMatchingPursuit'].predict(X_test))
)

submission = pd.concat([test_data.Id, pd.Series(predictions, name='SalePrice')], axis=1)
submission.to_csv('data/submission.csv', index=False)
print('dumped submission.csv into the current folder')
