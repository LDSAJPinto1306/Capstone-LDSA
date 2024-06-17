# custom_transformers.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateTimeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self    
    
    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            if column in X.columns:
                X[column] = pd.to_datetime(X[column], errors='coerce')
            else:
                print(f"Warning: '{column}' does not exist in the DataFrame.")
        return X



class CalculateJailTime(BaseEstimator, TransformerMixin):
    """
    A custom transformer to convert specified columns of a DataFrame to lowercase.
    """
    
    def fit(self, X, y=None):
        # Nothing to do here as there's no fitting process for lowering case
        return self    
    
    def transform(self, X):
            X = X.copy()
            if 'c_jail_out' in X.columns and 'c_jail_in' in X.columns:
                X['c_jail_time'] = (X['c_jail_out'] - X['c_jail_in']).dt.days
            else:
                print("Warning: Required columns for calculating jail time do not exist in the DataFrame.")
            return X
        
        
        