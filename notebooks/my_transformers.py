from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


class DenseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array(X.todense())


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.log_transformer = None

    def fit(self, X, y=None):
        self.log_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('log_transform', FunctionTransformer(np.log1p, validate=False)),
            ('scaler', StandardScaler())
        ])
        self.log_transformer.fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        X_new[self.columns] = self.log_transformer.transform(X[self.columns])
        return X_new


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, log_feat, num_feat, cat_feat):
        self.log_feat = log_feat
        self.num_feat = num_feat
        self.cat_feat = cat_feat
        self.preprocessor = None

    def fit(self, X, y=None):
        log_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('log_transform', FunctionTransformer(np.log1p, validate=False)),
            ('scaler', StandardScaler())
        ])

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ('dense', DenseTransformer()),
            ('pca', PCA(n_components=2))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('log', log_pipeline, self.log_feat),
            ('num', num_pipeline, self.num_feat),
            ('cat', cat_pipeline, self.cat_feat)
        ])

        self.preprocessor.fit(X)
        return self

    def transform(self, X, y=None):
        return self.preprocessor.transform(X)


class MyModel(BaseEstimator):
    def __init__(self):
        self.pipeline = None

    def fit(self, X, y=None):
        log_feat = ['LoanAmount', 'Total_Income']
        num_feat = ['Loan_Amount_Term', 'Credit_History']
        cat_feat = X.dtypes[X.dtypes == 'object'].index.tolist()

        self.pipeline = Pipeline([
            ('preprocessor', Preprocessor(log_feat, num_feat, cat_feat)),
            ('clf', SVC(C=0.1, gamma='scale', kernel='sigmoid', random_state=42))
        ])

        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
