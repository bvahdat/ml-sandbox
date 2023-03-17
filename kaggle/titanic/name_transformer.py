from sklearn.base import BaseEstimator, TransformerMixin

class NameTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.Name = X.Name.str.extract('([a-zA-Z]+)\.', expand=False)
        X.Name = X.Name.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Infrequent')
        X.Name = X.Name.replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')

        return X