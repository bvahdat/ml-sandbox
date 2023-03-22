# How to run

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment:

```
conda create -n titanic -c conda-forge pandas scikit-learn python=3.10 
conda activate titanic
python titanic.py
```

Which yields an output similar to:

```
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
accuracy of the model ExtraTreesClassifier: 0.795 (+/- 0.021)
accuracy of the model GaussianNB: 0.799 (+/- 0.034)
accuracy of the model LogisticRegression: 0.829 (+/- 0.028)
accuracy of the model SVC: 0.833 (+/- 0.026)
best model score is through: SVC with a mean accuracy of: 0.833
best grid searched estimator is: SVC(C=1, gamma=0.1) with a mean cross-validated score of 0.833 and params {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}
dumped submission.csv into the current folder

```

Finally you can remove the created environment above if you want:

```
conda remove --name titanic --all
```

The currently achieved accuracy is around `.78229`.
