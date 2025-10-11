"""
Regression algorithms for stock prediction
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .arima_regression import ARIMARegression

__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'ARIMARegression'
]
