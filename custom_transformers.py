"""
Custom scikit-learn transformers used by the Heart Disease pipeline.

Kept in its own module so the saved pipeline (.pkl) can be loaded by both
train_model.py and app.py.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap outliers at Q1 - 1.5*IQR and Q3 + 1.5*IQR (limits learned from training data only)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
