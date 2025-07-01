import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col], errors='coerce')
        X['transaction_hour'] = X[self.time_col].dt.hour
        X['transaction_day'] = X[self.time_col].dt.day
        X['transaction_month'] = X[self.time_col].dt.month
        return X.drop(columns=[self.time_col])

class Aggregator(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId'):
        self.group_col = group_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        agg_df = X.groupby(self.group_col).agg({
            'Value': ['sum', 'mean', 'count', 'std'],
            'Amount': ['mean', 'std']
        })
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df.reset_index(inplace=True)
        return agg_df

def build_pipeline(numerical_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('time_features', TimeFeatures()),
        ('preprocessor', preprocessor)
    ])

    return pipeline
