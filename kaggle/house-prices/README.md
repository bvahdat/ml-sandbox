# What is this codebase about?

See [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

# How to run

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment:

```
conda create -n house-prices -c conda-forge catboost==1.2.2 lightgbm==4.3.0 pandas==2.2.0 pyarrow==15.0.0 scikit-learn==1.4.0 python=3.10.13
conda activate house-prices
python house-prices.py
```

Which yields an output similar to:

```
catboost version: 1.2.2
lightgbm version: 4.3.0
pandas version: 2.2.0
pyarrow version: 15.0.0
sklearn version: 1.4.0
fixing the skewness of the feature LotFrontage: 1.352361873402816 => 0.9355415599396448
fixing the skewness of the feature LotArea: 12.822431401556724 => 0.50475055129892
skip fixing the skewness of the feature OverallCond: 0.5703120502855311 => 0.7461828792022595
skip fixing the skewness of the feature YearBuilt: 0.5998055475020266 => 0.6262153888824475
fixing the skewness of the feature MasVnrArea: 2.5993529487505938 => 0.5091273771059409
fixing the skewness of the feature BsmtFinSF1: 1.4254209404979759 => 0.6178169973982822
fixing the skewness of the feature BsmtFinSF2: 4.146110709722618 => 2.4589957247646588
skip fixing the skewness of the feature BsmtUnfSF: 0.9189375084031303 => 2.1591687996006197
skip fixing the skewness of the feature TotalBsmtSF: 1.1627634332049621 => 4.981681454995597
fixing the skewness of the feature 1stFlrSF: 1.4696044169256821 => 0.06482768155842679
fixing the skewness of the feature 2ndFlrSF: 0.8616747488436027 => 0.30504921600877727
fixing the skewness of the feature LowQualFinSF: 12.088761003370664 => 8.55769041866321
fixing the skewness of the feature GrLivArea: 1.269357688230336 => 0.013187581757008814
fixing the skewness of the feature BsmtFullBath: 0.6244509502363199 => 0.42453241374269063
fixing the skewness of the feature BsmtHalfBath: 3.9315938391525584 => 3.774762007425795
fixing the skewness of the feature HalfBath: 0.6945664946629632 => 0.5806915253872976
fixing the skewness of the feature KitchenAbvGr: 4.302254369609591 => 3.520350639382798
fixing the skewness of the feature TotRmsAbvGrd: 0.7583669060998621 => 0.035106990669190784
fixing the skewness of the feature Fireplaces: 0.7334945989608231 => 0.2375873474793804
fixing the skewness of the feature WoodDeckSF: 1.8424328111184782 => 0.1580330005981843
fixing the skewness of the feature OpenPorchSF: 2.5351137294802557 => 0.041797304052571066
fixing the skewness of the feature EnclosedPorch: 4.003891220540856 => 1.961080589820378
fixing the skewness of the feature 3SsnPorch: 11.376064682827481 => 8.825255765053772
fixing the skewness of the feature ScreenPorch: 3.9466937029936977 => 2.9459051310687494
fixing the skewness of the feature PoolArea: 16.89832791614449 => 14.99833472413935
fixing the skewness of the feature MiscVal: 21.9471948077491 => 5.213983641404962
skip fixing the skewness of the feature MoSold: 0.7549550584404429 => 4.755543201058093
fixing the skewness of the feature SqFtPerRoom: 0.8939915974266288 => 0.20115110154947938
skip fixing the skewness of the feature Total_Home_Quality: 0.5641233921894949 => 1.7448580941784118
fixing the skewness of the feature HighQualSF: 1.2520356731990414 => 0.0013318112480811782
MSE of the model BayesianRidge: 0.017 using the params: ({'alpha_1': 4.292637345887363e-06, 'alpha_2': 3.5910479175977224e-05, 'lambda_1': 1.12397859959266e-05, 'lambda_2': 6.448037694757948e-06, 'max_iter': 225, 'tol': 0.0013377754260487595})
MSE of the model CatBoostRegressor: 0.014 using the params: ({'depth': 4, 'iterations': 6758, 'learning_rate': 0.005963014595584123})
MSE of the model ExtraTreesRegressor: 0.018 using the params: ({'n_estimators': 171})
MSE of the model GradientBoostingRegressor: 0.015 using the params: ({'learning_rate': 0.10741679115723948, 'n_estimators': 231})
MSE of the model LGBMRegressor: 0.016 using the params: ({'learning_rate': 0.15248233985506102, 'max_depth': 2, 'n_estimators': 298, 'num_leaves': 42})
dumped data/submission.csv
```

Finally the created environment above can be removed through:

```
conda deactivate && conda remove --name house-prices --all
```

The achieved `MSE` score is `.12219`.
