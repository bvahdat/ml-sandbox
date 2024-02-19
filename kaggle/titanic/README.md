# What is this codebase about?

See [here](https://www.kaggle.com/c/titanic).

# How to run

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment:

```
conda create -n titanic -c conda-forge pandas==2.2.0 pyarrow==15.0.0 scikit-learn==1.4.0 python=3.10.13
conda activate titanic
python titanic.py
```

Which yields an output similar to:

```
pandas version: 2.2.0
pyarrow version: 15.0.0
sklearn version: 1.4.0
feature skewness before transformation: 
 Age      0.389108
SibSp    3.695352
Parch    2.749117
Fare     4.787317
dtype: float64
feature skewness after transformation: 
 Age      0.000794
SibSp    3.695352
Parch    2.749117
Fare     0.000125
dtype: float64
feature skewness before transformation: 
 Age      0.441744
SibSp    4.168337
Parch    4.654462
Fare     3.691621
dtype: float64
feature skewness after transformation: 
 Age      0.062056
SibSp    4.168337
Parch    4.654462
Fare     0.700397
dtype: float64
accuracy of the model ExtraTreesClassifier: 0.795 (+/- 0.043)
accuracy of the model GaussianNB: 0.802 (+/- 0.040)
accuracy of the model LogisticRegression: 0.824 (+/- 0.027)
accuracy of the model SVC: 0.834 (+/- 0.026)
best model score is through: SVC with a mean accuracy of: 0.834
best grid searched estimator is: SVC(C=100, gamma=0.01) with a mean cross-validated score of 0.834 and params {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
dumped data/submission.csv
```

Finally the created environment above can be removed through:

```
conda remove --name titanic --all
```

The achieved accuracy score is `.78708`.
