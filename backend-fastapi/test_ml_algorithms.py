"""
Test file for all ML algorithms in the Stock Price Insight Arena
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all ML algorithms
from algorithms.classification.svm_classifier import SVMClassifier
from algorithms.classification.decision_tree_classifier import DecisionTreeClassifier
from algorithms.classification.random_forest_classifier import RandomForestClassifier
from algorithms.regression.linear_regression import LinearRegression
from algorithms.regression.logistic_regression import LogisticRegression
from algorithms.regression.arima_regression import ARIMARegression
from algorithms.clustering.kmeans_clustering import KMeansClustering
from algorithms.clustering.tsne_clustering import TSNEClustering
from algorithms.neural_networks.ann_regressor import ANNRegressor
from algorithms.neural_networks.cnn_regressor import CNNRegressor
from algorithms.other.naive_bayes_classifier import NaiveBayesClassifier
from algorithms.other.knn_classifier import KNNClassifier
from algorithms.other.lazy_learning import LazyLearning


def create_sample_stock_data(n_days=100):
    """Create sample stock data for testing"""
    np.random.seed(42)
    
    # Generate realistic stock price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
    
    prices = [base_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    # Generate volume data
    base_volume = 1000000
    volume = np.random.normal(base_volume, base_volume * 0.3, n_days)
    volume = np.maximum(volume, 10000)  # Ensure positive volume
    
    # Create DataFrame
    dates = [datetime.now() - timedelta(days=i) for i in range(n_days, 0, -1)]
    
    data = []
    for i, (date, price, vol) in enumerate(zip(dates, prices[1:], volume)):
        # Generate high/low based on price
        daily_volatility = abs(np.random.normal(0, 0.01))
        high = price * (1 + daily_volatility)
        low = price * (1 - daily_volatility)
        
        data.append({
            'date': date,
            'close': price,
            'volume': vol,
            'high': high,
            'low': low,
            'open': prices[i] if i < len(prices) - 1 else price
        })
    
    return pd.DataFrame(data)


class TestClassificationAlgorithms:
    """Test classification algorithms"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = create_sample_stock_data(200)
    
    def test_svm_classifier(self):
        """Test Support Vector Machine Classifier"""
        model = SVMClassifier(kernel='rbf', C=1.0, random_state=42)
        
        # Test training
        results = model.train(self.df)
        
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        assert 'n_support_vectors' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 'probability_up' in prediction
        assert 'probability_down' in prediction
        assert 'decision_function' in prediction
        assert prediction['prediction'] in ['up', 'down']
    
    def test_decision_tree_classifier(self):
        """Test Decision Tree Classifier"""
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        
        # Test training
        results = model.train(self.df)
        
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        assert 'feature_importance' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 'probability_up' in prediction
        assert 'probability_down' in prediction
        assert prediction['prediction'] in ['up', 'down']
    
    def test_random_forest_classifier(self):
        """Test Random Forest Classifier"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Test training
        results = model.train(self.df)
        
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'feature_importance' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert prediction['prediction'] in ['up', 'down']


class TestRegressionAlgorithms:
    """Test regression algorithms"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = create_sample_stock_data(200)
    
    def test_linear_regression(self):
        """Test Linear Regression"""
        model = LinearRegression(random_state=42)
        
        # Test training
        results = model.train(self.df)
        
        assert 'mse' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'r2_score' in results
        assert 'coefficients' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'predicted_price' in prediction
        assert 'current_price' in prediction
        assert 'price_change' in prediction
        assert 'direction' in prediction
        assert prediction['direction'] in ['up', 'down']
    
    def test_logistic_regression(self):
        """Test Logistic Regression"""
        model = LogisticRegression(random_state=42)
        
        # Test training
        results = model.train(self.df)
        
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'coefficients' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert prediction['prediction'] in ['up', 'down']
    
    def test_arima_regression(self):
        """Test ARIMA Regression"""
        model = ARIMARegression(order=(1, 1, 1), auto_arima=False)
        
        # Test training
        results = model.train(self.df)
        
        assert 'mse' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'aic' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'predicted_price' in prediction
        assert 'current_price' in prediction
        assert 'direction' in prediction
        assert prediction['direction'] in ['up', 'down']


class TestClusteringAlgorithms:
    """Test clustering algorithms"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = create_sample_stock_data(200)
    
    def test_kmeans_clustering(self):
        """Test K-Means Clustering"""
        model = KMeansClustering(n_clusters=3, random_state=42)
        
        # Test training
        results = model.fit(self.df)
        
        assert 'silhouette_score' in results
        assert 'cluster_centers' in results
        assert 'cluster_analysis' in results
        assert model.is_fitted
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'predicted_cluster' in prediction
        assert 'cluster_distance' in prediction
        assert 'cluster_characteristics' in prediction
    
    def test_tsne_clustering(self):
        """Test t-SNE Clustering"""
        model = TSNEClustering(n_components=2, random_state=42)
        
        # Test training
        results = model.fit_transform(self.df, n_clusters=3)
        
        assert 'embedded_data' in results
        assert 'cluster_labels' in results
        assert 'silhouette_score' in results
        assert model.is_fitted


class TestNeuralNetworkAlgorithms:
    """Test neural network algorithms"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = create_sample_stock_data(200)
    
    def test_ann_regressor(self):
        """Test Artificial Neural Network"""
        model = ANNRegressor(hidden_layers=[32, 16], epochs=5, random_state=42)
        
        # Test training
        results = model.train(self.df)
        
        assert 'mse' in results
        assert 'rmse' in results
        assert 'r2_score' in results
        assert 'training_history' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'predicted_price' in prediction
        assert 'current_price' in prediction
        assert 'direction' in prediction
        assert prediction['direction'] in ['up', 'down']
    
    def test_cnn_regressor(self):
        """Test Convolutional Neural Network"""
        try:
            model = CNNRegressor(sequence_length=10, epochs=5, random_state=42)
            
            # Test training
            results = model.train(self.df)
            
            assert 'mse' in results
            assert 'rmse' in results
            assert 'r2_score' in results
            assert model.is_trained
            
            # Test prediction
            prediction = model.predict(self.df)
            
            assert 'predicted_price' in prediction
            assert 'current_price' in prediction
            assert 'direction' in prediction
            assert prediction['direction'] in ['up', 'down']
            
        except ImportError:
            pytest.skip("TensorFlow not available - CNN test skipped")


class TestOtherAlgorithms:
    """Test other algorithms"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = create_sample_stock_data(200)
    
    def test_naive_bayes_classifier(self):
        """Test Naive Bayes Classifier"""
        model = NaiveBayesClassifier(random_state=42)
        
        # Test training
        results = model.train(self.df)
        
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'class_probabilities' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert prediction['prediction'] in ['up', 'down']
    
    def test_knn_classifier(self):
        """Test K-Nearest Neighbors Classifier"""
        model = KNNClassifier(n_neighbors=5, random_state=42)
        
        # Test training
        results = model.train(self.df)
        
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'cv_scores' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 'neighbor_distances' in prediction
        assert prediction['prediction'] in ['up', 'down']
    
    def test_lazy_learning(self):
        """Test Lazy Learning"""
        model = LazyLearning(n_neighbors=5, random_state=42)
        
        # Test training
        results = model.train(self.df)
        
        assert 'training_samples' in results
        assert 'lazy_learning' in results
        assert model.is_trained
        
        # Test prediction
        prediction = model.predict(self.df)
        
        assert 'predicted_price' in prediction
        assert 'current_price' in prediction
        assert 'neighbor_analysis' in prediction
        assert 'lazy_computation' in prediction


class TestAlgorithmIntegration:
    """Test algorithm integration and edge cases"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = create_sample_stock_data(200)
        self.small_df = create_sample_stock_data(20)  # Small dataset for edge cases
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        # Test with very small dataset
        model = DecisionTreeClassifier()
        
        try:
            results = model.train(self.small_df)
            # Should either work or raise a clear error
            assert 'accuracy' in results or 'error' in str(results)
        except ValueError as e:
            assert "insufficient" in str(e).lower() or "no valid data" in str(e).lower()
    
    def test_feature_preparation(self):
        """Test feature preparation across algorithms"""
        model = LinearRegression()
        
        # Test feature preparation
        X, y = model.prepare_features(self.df)
        
        assert X.shape[0] > 0
        assert y.shape[0] > 0
        assert X.shape[0] == y.shape[0]
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
    
    def test_model_info_retrieval(self):
        """Test model information retrieval"""
        model = RandomForestClassifier()
        model.train(self.df)
        
        info = model.get_model_info()
        
        assert 'model_type' in info
        assert 'is_trained' in info
        assert info['is_trained'] == True
    
    def test_prediction_consistency(self):
        """Test prediction consistency across multiple calls"""
        model = DecisionTreeClassifier(random_state=42)
        model.train(self.df)
        
        # Make multiple predictions
        pred1 = model.predict(self.df)
        pred2 = model.predict(self.df)
        
        # Should be consistent
        assert pred1['prediction'] == pred2['prediction']
        assert pred1['confidence'] == pred2['confidence']


def run_all_tests():
    """Run all tests"""
    print("Running ML Algorithms Tests...")
    print("=" * 50)
    
    # Test data creation
    print("Creating test data...")
    df = create_sample_stock_data(100)
    print(f"Created test data with {len(df)} rows")
    
    # Test each algorithm category
    test_classes = [
        TestClassificationAlgorithms,
        TestRegressionAlgorithms,
        TestClusteringAlgorithms,
        TestNeuralNetworkAlgorithms,
        TestOtherAlgorithms,
        TestAlgorithmIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        
        # Create test instance
        test_instance = test_class()
        test_instance.setup_method()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
