"""
Neural network algorithms for stock prediction
"""

from .ann_regressor import ANNRegressor
from .cnn_regressor import CNNRegressor

__all__ = [
    'ANNRegressor',
    'CNNRegressor'
]
