from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np


class SkewedTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in X.columns:
            skewness_before = abs(X[column].skew())
            if (skewness_before >= .5):
                X[column] = np.log1p(X[column])
                skewness_after = abs(X[column].skew())
                print(f'fixing the skewness of the feature {column}: {skewness_before} => {skewness_after}')

        return X
