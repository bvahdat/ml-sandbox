from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
from scipy.stats import skew


class SkewedTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in X.columns:
            skewness_before = abs(skew(X[column]))
            if (skewness_before >= .5):
                log1p=np.log1p(X[column])
                skewness_after=abs(skew(log1p))
                if (skewness_after < skewness_before):
                    X[column]=log1p
                    print(f'fixing the skewness of the feature {column}: {skewness_before} => {skewness_after}')
                else:
                    print(f'skip fixing the skewness of the feature {column}: {skewness_before} => {skewness_after}')
        return X
