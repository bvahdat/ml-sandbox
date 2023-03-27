# What is this codebase about?

See [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

# How to run

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment:

```
conda create -n house-prices -c conda-forge catboost lightgbm pandas scikit-learn python=3.10 
conda activate house-prices
python house-prices.py
```

Which yields an output similar to:

```
fixing the skewness of the feature LotFrontage: 1.3530572744614882 => 0.9360226268818653
fixing the skewness of the feature LotArea: 12.829024853018762 => 0.505010100221913
fixing the skewness of the feature OverallCond: 0.5706053117352524 => 0.7465665755888299
fixing the skewness of the feature YearBuilt: 0.6001139748696814 => 0.6265373965143912
fixing the skewness of the feature MasVnrArea: 2.600689567911466 => 0.5093891766464348
fixing the skewness of the feature BsmtFinSF1: 1.4261539094248366 => 0.6181346864743393
fixing the skewness of the feature BsmtFinSF2: 4.148242690691333 => 2.4602601705198235
fixing the skewness of the feature BsmtUnfSF: 0.9194100373384432 => 2.160279070674231
fixing the skewness of the feature TotalBsmtSF: 1.163361340420729 => 4.984243096687738
fixing the skewness of the feature 1stFlrSF: 1.4703601055379227 => 0.06486101674723506
fixing the skewness of the feature 2ndFlrSF: 0.8621178325657642 => 0.305206076056322
fixing the skewness of the feature LowQualFinSF: 12.094977192517302 => 8.56209088801536
fixing the skewness of the feature GrLivArea: 1.2700104075191514 => 0.013194362973261287
fixing the skewness of the feature BsmtFullBath: 0.6247720505722756 => 0.42475071351567556
fixing the skewness of the feature BsmtHalfBath: 3.9336155129159094 => 3.7767030363381187
fixing the skewness of the feature HalfBath: 0.6949236492716564 => 0.5809901240903949
fixing the skewness of the feature KitchenAbvGr: 4.304466641562935 => 3.5221608468499483
fixing the skewness of the feature TotRmsAbvGrd: 0.7587568676624701 => 0.03512504311429316
fixing the skewness of the feature Fireplaces: 0.733871770878103 => 0.2377095178068746
fixing the skewness of the feature WoodDeckSF: 1.8433802126628294 => 0.15811426310497517
fixing the skewness of the feature OpenPorchSF: 2.5364173160468444 => 0.04181879673885529
fixing the skewness of the feature EnclosedPorch: 4.005950070504265 => 1.9620890015074595
fixing the skewness of the feature 3SsnPorch: 11.381914394786643 => 8.829793819788062
fixing the skewness of the feature ScreenPorch: 3.948723141292199 => 2.9474199516113737
fixing the skewness of the feature PoolArea: 16.9070172435751 => 15.006047051771919
fixing the skewness of the feature MiscVal: 21.958480324447216 => 5.216664735729965
fixing the skewness of the feature SqFtPerRoom: 0.8944512988686179 => 0.20125453591251607
fixing the skewness of the feature Total_Home_Quality: 0.5644134713553347 => 1.7457553216068
fixing the skewness of the feature HighQualSF: 1.2526794852953163 => 0.0013324960817563075
MSE of the model BayesianRidge: 0.016 using the params: ({'alpha_1': 9.143366252323894e-05, 'alpha_2': 6.056412804338572e-05, 'lambda_1': 3.932602969444943e-05, 'lambda_2': 5.771580063805139e-06, 'n_iter': 343, 'tol': 0.0010141031057467208})
MSE of the model CatBoostRegressor: 0.014 using the params: ({'depth': 4, 'iterations': 6539, 'learning_rate': 0.006470201433377933})
MSE of the model ExtraTreesRegressor: 0.018 using the params: ({'n_estimators': 298})
MSE of the model GradientBoostingRegressor: 0.015 using the params: ({'learning_rate': 0.1236467550710634, 'n_estimators': 199})
MSE of the model LGBMRegressor: 0.016 using the params: ({'learning_rate': 0.1330158380805924, 'max_depth': 2, 'n_estimators': 290, 'num_leaves': 46})
dumped data/submission.csv
```

Finally the created environment above can be removed through:

```
conda remove --name house-prices --all
```

The achieved `RMSE` score is `.12329`.
