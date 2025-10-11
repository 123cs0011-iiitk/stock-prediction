# Machine Learning Algorithms for Stock Prediction

This directory contains implementations of various machine learning algorithms for stock price prediction and analysis in the Stock Price Insight Arena project.

## 📁 Directory Structure

```
algorithms/
├── classification/          # Classification algorithms
│   ├── svm_classifier.py
│   ├── decision_tree_classifier.py
│   └── random_forest_classifier.py
├── regression/              # Regression algorithms
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   └── arima_regression.py
├── clustering/              # Clustering algorithms
│   ├── kmeans_clustering.py
│   └── tsne_clustering.py
├── neural_networks/         # Neural network algorithms
│   ├── ann_regressor.py
│   └── cnn_regressor.py
├── other/                   # Other algorithms
│   ├── naive_bayes_classifier.py
│   ├── knn_classifier.py
│   └── lazy_learning.py
└── utils/                   # Utility functions
```

## 🤖 Available Algorithms

### Classification Algorithms
- **Support Vector Machine (SVM)**: SVM with kernel methods for stock direction prediction
- **Decision Tree Classifier**: Tree-based classification for stock direction prediction
- **Random Forest Classifier**: Ensemble of decision trees for robust classification

### Regression Algorithms
- **Linear Regression**: Linear relationship modeling for price prediction
- **Logistic Regression**: Logistic function for probability-based classification
- **ARIMA**: AutoRegressive Integrated Moving Average for time series forecasting

### Clustering Algorithms
- **K-Means Clustering**: Unsupervised clustering for market segmentation
- **t-SNE Clustering**: Dimensionality reduction and visualization for stock data

### Neural Network Algorithms
- **Artificial Neural Network (ANN)**: Deep learning for complex pattern recognition
- **Convolutional Neural Network (CNN)**: CNN for time series stock data analysis

### Other Algorithms
- **Naive Bayes Classifier**: Probabilistic classifier based on Bayes' theorem
- **K-Nearest Neighbors (K-NN)**: Instance-based learning for classification
- **Lazy Learning**: Deferred computation until prediction time

## 🚀 Quick Start

### 1. Training an Algorithm

```python
from algorithms.classification.decision_tree_classifier import DecisionTreeClassifier
import pandas as pd

# Load your stock data
df = pd.read_csv('stock_data.csv')

# Initialize and train the model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
results = model.train(df)

print(f"Training accuracy: {results['accuracy']:.4f}")
```

### 2. Making Predictions

```python
# Make predictions
prediction = model.predict(df)

print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.4f}")
print(f"Probability Up: {prediction['probability_up']:.4f}")
```

### 3. Using FastAPI Endpoints

```bash
# Train a model
curl -X POST "http://localhost:8000/api/v1/algorithms/classification/decision-tree/train?symbol=AAPL&max_depth=5"

# Get available algorithms
curl -X GET "http://localhost:8000/api/v1/algorithms/available"
```

## 📊 Features

### Technical Indicators
All algorithms automatically calculate and use the following technical indicators:

- **Price-based features**: Price changes, moving averages, ratios
- **Volatility features**: Rolling standard deviations
- **Volume features**: Volume changes and ratios
- **Technical indicators**: RSI, MACD, Bollinger Bands
- **High-Low features**: Price position within daily range

### Model Evaluation
Each algorithm provides comprehensive evaluation metrics:

- **Classification**: Accuracy, precision, recall, F1-score, confusion matrix
- **Regression**: MSE, RMSE, MAE, R² score, MAPE
- **Clustering**: Silhouette score, Calinski-Harabasz score, Davies-Bouldin score

### Feature Importance
Most algorithms provide feature importance analysis:

- **Decision Trees**: Feature importance scores
- **Random Forest**: Feature importance with ensemble voting
- **Linear/Logistic Regression**: Coefficient analysis

## 🔧 Configuration

### Algorithm Parameters

#### Support Vector Machine Classifier
```python
model = SVMClassifier(
    kernel='rbf',           # Kernel type (linear, poly, rbf, sigmoid)
    C=1.0,                  # Regularization parameter
    gamma='scale',          # Kernel coefficient
    degree=3,               # Degree for polynomial kernel
    probability=True,       # Enable probability estimates
    random_state=42
)
```

#### Decision Tree Classifier
```python
model = DecisionTreeClassifier(
    max_depth=10,           # Maximum tree depth
    min_samples_split=2,    # Minimum samples to split
    min_samples_leaf=1,     # Minimum samples per leaf
    random_state=42
)
```

#### Random Forest Classifier
```python
model = RandomForestClassifier(
    n_estimators=100,       # Number of trees
    max_depth=None,         # Maximum tree depth
    min_samples_split=2,    # Minimum samples to split
    random_state=42
)
```

#### ARIMA Regression
```python
model = ARIMARegression(
    order=(1, 1, 1),        # (p, d, q) parameters
    seasonal_order=(0, 0, 0, 0),  # Seasonal parameters
    auto_arima=True         # Auto-select best parameters
)
```

#### Neural Networks
```python
# ANN
model = ANNRegressor(
    hidden_layers=[64, 32, 16],  # Hidden layer sizes
    activation='relu',            # Activation function
    dropout_rate=0.2,            # Dropout rate
    learning_rate=0.001,         # Learning rate
    epochs=100                   # Training epochs
)

# CNN
model = CNNRegressor(
    sequence_length=30,          # Time series sequence length
    filters=64,                  # Number of filters
    kernel_size=3,               # Convolution kernel size
    hidden_layers=[32, 16],      # Dense layer sizes
    epochs=100
)
```

## 📈 Performance Guidelines

### Data Requirements
- **Minimum data**: At least 50-100 data points for reliable results
- **Recommended data**: 200+ data points for optimal performance
- **Data quality**: Clean data without excessive missing values

### Algorithm Selection

#### For Classification (Up/Down Prediction)
1. **Quick results**: Naive Bayes or Logistic Regression
2. **Balanced performance**: Decision Tree or K-NN
3. **High accuracy**: SVM, Random Forest or ANN
4. **High-dimensional data**: SVM with RBF kernel

#### For Price Prediction
1. **Linear trends**: Linear Regression
2. **Time series**: ARIMA
3. **Complex patterns**: ANN or CNN

#### For Market Analysis
1. **Market segmentation**: K-Means
2. **Data visualization**: t-SNE
3. **Pattern discovery**: Lazy Learning

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd backend-fastapi
python test_ml_algorithms.py
```

The test suite includes:
- Algorithm training and prediction tests
- Edge case handling (insufficient data, etc.)
- Performance validation
- Integration tests

## 🔍 Monitoring and Debugging

### Model Information
```python
# Get model details
info = model.get_model_info()
print(f"Model type: {info['model_type']}")
print(f"Parameters: {info}")
```

### Feature Analysis
```python
# Get feature importance (where available)
importance = model.get_feature_importance()
print("Feature importance:", importance)
```

### Training History (Neural Networks)
```python
# Access training history
history = results['training_history']
print("Training loss:", history['loss'][-1])
print("Validation loss:", history['val_loss'][-1])
```

## 🚨 Error Handling

All algorithms include robust error handling:

- **Data validation**: Checks for sufficient data and valid features
- **Parameter validation**: Validates algorithm parameters
- **Graceful failures**: Clear error messages for debugging

Common error scenarios:
- Insufficient training data
- Invalid algorithm parameters
- Missing required features
- TensorFlow availability (for neural networks)

## 📚 Dependencies

### Core Dependencies
- `scikit-learn`: Machine learning algorithms
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `statsmodels`: Statistical models (ARIMA)

### Optional Dependencies
- `tensorflow`: Neural networks (ANN, CNN)
- `matplotlib`: Visualization
- `seaborn`: Statistical visualization

## 🔄 Updates and Maintenance

### Adding New Algorithms
1. Create algorithm file in appropriate category directory
2. Implement standard interface (train, predict, get_model_info)
3. Add FastAPI routes in `ml_algorithms_routes.py`
4. Update tests in `test_ml_algorithms.py`
5. Update this README

### Performance Optimization
- Use appropriate data types (float32 for neural networks)
- Implement early stopping for neural networks
- Cache feature calculations for repeated use
- Use parallel processing where applicable

## 📞 Support

For issues or questions:
1. Check the test suite for usage examples
2. Review algorithm-specific documentation
3. Check FastAPI endpoint documentation at `/docs`
4. Examine error messages for debugging guidance

## 🎯 Best Practices

1. **Data Preprocessing**: Ensure clean, normalized data
2. **Cross-validation**: Use proper train/validation/test splits
3. **Hyperparameter Tuning**: Experiment with different parameters
4. **Ensemble Methods**: Combine multiple algorithms for better results
5. **Regular Retraining**: Update models with fresh data
6. **Performance Monitoring**: Track model performance over time

## 📊 Example Results

### Classification Results
```
Decision Tree Accuracy: 0.7234
Random Forest Accuracy: 0.7891
Naive Bayes Accuracy: 0.6543
```

### Regression Results
```
Linear Regression R²: 0.8234
ARIMA RMSE: 2.3456
ANN R²: 0.8765
```

### Clustering Results
```
K-Means Silhouette Score: 0.4567
t-SNE Visualization: 2D embedding completed
```

This comprehensive ML algorithms suite provides powerful tools for stock market analysis and prediction, with each algorithm optimized for specific use cases and market conditions.
