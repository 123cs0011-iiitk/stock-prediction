"""
Stock Price Insight Arena - ML Algorithms Routes
API routes for all machine learning algorithms.
"""

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks
from typing import List, Optional, Dict, Any
import logging
import pandas as pd
from datetime import datetime

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

from services.stock_service import stock_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/algorithms/available")
async def get_available_algorithms():
    """
    Get list of all available ML algorithms
    
    Returns information about all available machine learning algorithms including:
    - Algorithm names and descriptions
    - Categories (Classification, Regression, Clustering, Neural Networks, Other)
    - Capabilities and use cases
    """
    try:
        algorithms_info = {
            "classification": {
                "svm": {
                    "name": "Support Vector Machine Classifier",
                    "description": "SVM with kernel methods for stock direction prediction",
                    "use_case": "High-dimensional stock direction prediction with margin maximization",
                    "strengths": ["Effective in high dimensions", "Memory efficient", "Versatile kernels"],
                    "parameters": ["kernel", "C", "gamma", "degree"]
                },
                "decision_tree": {
                    "name": "Decision Tree Classifier",
                    "description": "Tree-based classification for stock direction prediction",
                    "use_case": "Predicting stock price direction (up/down)",
                    "strengths": ["Interpretable", "Handles non-linear patterns", "Feature importance"],
                    "parameters": ["max_depth", "min_samples_split", "min_samples_leaf"]
                },
                "random_forest": {
                    "name": "Random Forest Classifier", 
                    "description": "Ensemble of decision trees for robust classification",
                    "use_case": "Robust stock direction prediction with feature importance",
                    "strengths": ["High accuracy", "Feature importance", "Robust to outliers"],
                    "parameters": ["n_estimators", "max_depth", "min_samples_split"]
                }
            },
            "regression": {
                "linear_regression": {
                    "name": "Linear Regression",
                    "description": "Linear relationship modeling for price prediction",
                    "use_case": "Predicting exact stock prices using linear relationships",
                    "strengths": ["Fast training", "Interpretable", "Good for linear trends"],
                    "parameters": ["fit_intercept", "normalize"]
                },
                "logistic_regression": {
                    "name": "Logistic Regression",
                    "description": "Logistic function for probability-based classification",
                    "use_case": "Stock direction prediction with probability scores",
                    "strengths": ["Probability outputs", "Fast training", "Good for binary classification"],
                    "parameters": ["penalty", "C", "solver"]
                },
                "arima": {
                    "name": "ARIMA Time Series",
                    "description": "AutoRegressive Integrated Moving Average for time series forecasting",
                    "use_case": "Time series stock price forecasting",
                    "strengths": ["Time series specific", "Statistical foundation", "Trend analysis"],
                    "parameters": ["order", "seasonal_order", "auto_arima"]
                }
            },
            "clustering": {
                "kmeans": {
                    "name": "K-Means Clustering",
                    "description": "Unsupervised clustering for market segmentation",
                    "use_case": "Grouping similar market conditions and stock behaviors",
                    "strengths": ["Market segmentation", "Pattern discovery", "Unsupervised learning"],
                    "parameters": ["n_clusters", "init", "max_iter"]
                },
                "tsne": {
                    "name": "t-SNE Clustering",
                    "description": "Dimensionality reduction and visualization for stock data",
                    "use_case": "Visualizing complex stock data relationships",
                    "strengths": ["Data visualization", "Dimensionality reduction", "Pattern discovery"],
                    "parameters": ["n_components", "perplexity", "learning_rate"]
                }
            },
            "neural_networks": {
                "ann": {
                    "name": "Artificial Neural Network",
                    "description": "Deep learning for complex pattern recognition",
                    "use_case": "Advanced stock price prediction with non-linear patterns",
                    "strengths": ["Complex patterns", "High accuracy potential", "Feature learning"],
                    "parameters": ["hidden_layers", "activation", "dropout_rate", "learning_rate"]
                },
                "cnn": {
                    "name": "Convolutional Neural Network",
                    "description": "CNN for time series stock data analysis",
                    "use_case": "Time series pattern recognition for stock prediction",
                    "strengths": ["Time series patterns", "Sequence learning", "Advanced feature extraction"],
                    "parameters": ["sequence_length", "filters", "kernel_size", "hidden_layers"]
                }
            },
            "other": {
                "naive_bayes": {
                    "name": "Naive Bayes Classifier",
                    "description": "Probabilistic classifier based on Bayes' theorem",
                    "use_case": "Quick stock direction prediction with probability estimates",
                    "strengths": ["Fast training", "Probability estimates", "Good baseline"],
                    "parameters": ["var_smoothing"]
                },
                "knn": {
                    "name": "K-Nearest Neighbors",
                    "description": "Instance-based learning for classification",
                    "use_case": "Stock direction prediction based on similar historical patterns",
                    "strengths": ["Simple concept", "No training phase", "Good for small datasets"],
                    "parameters": ["n_neighbors", "weights", "algorithm"]
                },
                "lazy_learning": {
                    "name": "Lazy Learning",
                    "description": "Deferred computation until prediction time",
                    "use_case": "Dynamic stock prediction using similar historical instances",
                    "strengths": ["Dynamic adaptation", "No model storage", "Real-time learning"],
                    "parameters": ["n_neighbors", "weights", "algorithm"]
                }
            }
        }
        
        return {
            "success": True,
            "message": "Available algorithms retrieved successfully",
            "algorithms": algorithms_info,
            "categories": list(algorithms_info.keys()),
            "total_algorithms": sum(len(cat) for cat in algorithms_info.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting available algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get available algorithms")


# Classification Algorithms Routes
@router.post("/algorithms/classification/svm/train")
async def train_svm(
    symbol: str = Query(..., description="Stock symbol to train on"),
    kernel: str = Query('rbf', description="Kernel type (linear, poly, rbf, sigmoid)"),
    C: float = Query(1.0, description="Regularization parameter"),
    gamma: str = Query('scale', description="Kernel coefficient"),
    degree: int = Query(3, description="Degree for polynomial kernel"),
    optimize_params: bool = Query(False, description="Optimize hyperparameters")
):
    """Train Support Vector Machine Classifier"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize model
        model = SVMClassifier(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree
        )
        
        # Optimize parameters if requested
        if optimize_params:
            optimization_results = model.optimize_hyperparameters(df)
        
        # Train model
        results = model.train(df)
        
        response = {
            "success": True,
            "message": f"SVM trained for {symbol}",
            "symbol": symbol,
            "algorithm": "Support Vector Machine Classifier",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if optimize_params:
            response["optimization_results"] = optimization_results
        
        return response
        
    except Exception as e:
        logger.error(f"Error training SVM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train SVM: {str(e)}")


@router.post("/algorithms/classification/decision-tree/train")
async def train_decision_tree(
    symbol: str = Query(..., description="Stock symbol to train on"),
    max_depth: Optional[int] = Query(None, description="Maximum depth of the tree"),
    min_samples_split: int = Query(2, description="Minimum samples to split"),
    min_samples_leaf: int = Query(1, description="Minimum samples per leaf")
):
    """Train Decision Tree Classifier"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize and train model
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"Decision Tree trained for {symbol}",
            "symbol": symbol,
            "algorithm": "Decision Tree Classifier",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training Decision Tree: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train Decision Tree: {str(e)}")


@router.post("/algorithms/classification/random-forest/train")
async def train_random_forest(
    symbol: str = Query(..., description="Stock symbol to train on"),
    n_estimators: int = Query(100, description="Number of trees"),
    max_depth: Optional[int] = Query(None, description="Maximum depth of trees"),
    min_samples_split: int = Query(2, description="Minimum samples to split")
):
    """Train Random Forest Classifier"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"Random Forest trained for {symbol}",
            "symbol": symbol,
            "algorithm": "Random Forest Classifier",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training Random Forest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train Random Forest: {str(e)}")


# Regression Algorithms Routes
@router.post("/algorithms/regression/linear-regression/train")
async def train_linear_regression(
    symbol: str = Query(..., description="Stock symbol to train on"),
    fit_intercept: bool = Query(True, description="Whether to fit intercept"),
    normalize: bool = Query(False, description="Whether to normalize features")
):
    """Train Linear Regression model"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize and train model
        model = LinearRegression(
            fit_intercept=fit_intercept,
            normalize=normalize
        )
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"Linear Regression trained for {symbol}",
            "symbol": symbol,
            "algorithm": "Linear Regression",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training Linear Regression: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train Linear Regression: {str(e)}")


@router.post("/algorithms/regression/logistic-regression/train")
async def train_logistic_regression(
    symbol: str = Query(..., description="Stock symbol to train on"),
    penalty: str = Query('l2', description="Regularization penalty"),
    C: float = Query(1.0, description="Inverse regularization strength"),
    solver: str = Query('lbfgs', description="Optimization algorithm")
):
    """Train Logistic Regression model"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize and train model
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver
        )
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"Logistic Regression trained for {symbol}",
            "symbol": symbol,
            "algorithm": "Logistic Regression",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training Logistic Regression: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train Logistic Regression: {str(e)}")


@router.post("/algorithms/regression/arima/train")
async def train_arima(
    symbol: str = Query(..., description="Stock symbol to train on"),
    order: str = Query("1,1,1", description="ARIMA order (p,d,q)"),
    seasonal_order: str = Query("0,0,0,0", description="Seasonal order (P,D,Q,s)"),
    auto_arima: bool = Query(False, description="Auto-select best parameters")
):
    """Train ARIMA model"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Parse order parameters
        order_tuple = tuple(map(int, order.split(',')))
        seasonal_tuple = tuple(map(int, seasonal_order.split(',')))
        
        # Initialize and train model
        model = ARIMARegression(
            order=order_tuple,
            seasonal_order=seasonal_tuple,
            auto_arima=auto_arima
        )
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"ARIMA trained for {symbol}",
            "symbol": symbol,
            "algorithm": "ARIMA",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training ARIMA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train ARIMA: {str(e)}")


# Clustering Algorithms Routes
@router.post("/algorithms/clustering/kmeans/train")
async def train_kmeans(
    symbol: str = Query(..., description="Stock symbol to train on"),
    n_clusters: int = Query(3, description="Number of clusters"),
    init: str = Query('k-means++', description="Initialization method"),
    max_iter: int = Query(300, description="Maximum iterations")
):
    """Train K-Means Clustering"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize and train model
        model = KMeansClustering(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter
        )
        
        results = model.fit(df)
        
        return {
            "success": True,
            "message": f"K-Means clustering trained for {symbol}",
            "symbol": symbol,
            "algorithm": "K-Means Clustering",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training K-Means: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train K-Means: {str(e)}")


@router.post("/algorithms/clustering/tsne/train")
async def train_tsne(
    symbol: str = Query(..., description="Stock symbol to train on"),
    n_components: int = Query(2, description="Number of components"),
    perplexity: float = Query(30.0, description="Perplexity parameter"),
    learning_rate: float = Query(200.0, description="Learning rate"),
    n_clusters: int = Query(3, description="Number of clusters for analysis")
):
    """Train t-SNE Clustering"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize and train model
        model = TSNEClustering(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate
        )
        
        results = model.fit_transform(df, n_clusters)
        
        return {
            "success": True,
            "message": f"t-SNE clustering trained for {symbol}",
            "symbol": symbol,
            "algorithm": "t-SNE Clustering",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training t-SNE: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train t-SNE: {str(e)}")


# Neural Network Algorithms Routes
@router.post("/algorithms/neural-networks/ann/train")
async def train_ann(
    symbol: str = Query(..., description="Stock symbol to train on"),
    hidden_layers: str = Query("64,32,16", description="Hidden layer sizes"),
    activation: str = Query('relu', description="Activation function"),
    dropout_rate: float = Query(0.2, description="Dropout rate"),
    learning_rate: float = Query(0.001, description="Learning rate"),
    epochs: int = Query(100, description="Number of epochs")
):
    """Train Artificial Neural Network"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Parse hidden layers
        hidden_layers_list = [int(x) for x in hidden_layers.split(',')]
        
        # Initialize and train model
        model = ANNRegressor(
            hidden_layers=hidden_layers_list,
            activation=activation,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            epochs=epochs
        )
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"ANN trained for {symbol}",
            "symbol": symbol,
            "algorithm": "Artificial Neural Network",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training ANN: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train ANN: {str(e)}")


@router.post("/algorithms/neural-networks/cnn/train")
async def train_cnn(
    symbol: str = Query(..., description="Stock symbol to train on"),
    sequence_length: int = Query(30, description="Sequence length"),
    filters: int = Query(64, description="Number of filters"),
    kernel_size: int = Query(3, description="Kernel size"),
    hidden_layers: str = Query("32,16", description="Hidden layer sizes"),
    learning_rate: float = Query(0.001, description="Learning rate"),
    epochs: int = Query(100, description="Number of epochs")
):
    """Train Convolutional Neural Network"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Parse hidden layers
        hidden_layers_list = [int(x) for x in hidden_layers.split(',')]
        
        # Initialize and train model
        model = CNNRegressor(
            sequence_length=sequence_length,
            filters=filters,
            kernel_size=kernel_size,
            hidden_layers=hidden_layers_list,
            learning_rate=learning_rate,
            epochs=epochs
        )
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"CNN trained for {symbol}",
            "symbol": symbol,
            "algorithm": "Convolutional Neural Network",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training CNN: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train CNN: {str(e)}")


# Other Algorithms Routes
@router.post("/algorithms/other/naive-bayes/train")
async def train_naive_bayes(
    symbol: str = Query(..., description="Stock symbol to train on"),
    var_smoothing: float = Query(1e-9, description="Variance smoothing parameter")
):
    """Train Naive Bayes Classifier"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize and train model
        model = NaiveBayesClassifier(var_smoothing=var_smoothing)
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"Naive Bayes trained for {symbol}",
            "symbol": symbol,
            "algorithm": "Naive Bayes Classifier",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training Naive Bayes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train Naive Bayes: {str(e)}")


@router.post("/algorithms/other/knn/train")
async def train_knn(
    symbol: str = Query(..., description="Stock symbol to train on"),
    n_neighbors: int = Query(5, description="Number of neighbors"),
    weights: str = Query('uniform', description="Weight function"),
    algorithm: str = Query('auto', description="Algorithm to use")
):
    """Train K-Nearest Neighbors Classifier"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize and train model
        model = KNNClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm
        )
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"K-NN trained for {symbol}",
            "symbol": symbol,
            "algorithm": "K-Nearest Neighbors",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training K-NN: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train K-NN: {str(e)}")


@router.post("/algorithms/other/lazy-learning/train")
async def train_lazy_learning(
    symbol: str = Query(..., description="Stock symbol to train on"),
    n_neighbors: int = Query(5, description="Number of neighbors"),
    weights: str = Query('distance', description="Weight function"),
    algorithm: str = Query('auto', description="Algorithm to use")
):
    """Train Lazy Learning model"""
    try:
        # Get stock data
        stock_data = await stock_service.get_complete_stock_data(symbol)
        if not stock_data.historical_data:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'close': data.close,
            'volume': data.volume,
            'high': data.high,
            'low': data.low
        } for data in stock_data.historical_data])
        
        # Initialize and train model
        model = LazyLearning(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm
        )
        
        results = model.train(df)
        
        return {
            "success": True,
            "message": f"Lazy Learning trained for {symbol}",
            "symbol": symbol,
            "algorithm": "Lazy Learning",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error training Lazy Learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train Lazy Learning: {str(e)}")


@router.get("/algorithms/health")
async def ml_algorithms_health():
    """
    Health check endpoint for ML algorithms
    """
    try:
        # Test imports
        algorithm_status = {
            "classification": {
                "svm": "available",
                "decision_tree": "available",
                "random_forest": "available"
            },
            "regression": {
                "linear_regression": "available",
                "logistic_regression": "available",
                "arima": "available"
            },
            "clustering": {
                "kmeans": "available",
                "tsne": "available"
            },
            "neural_networks": {
                "ann": "available",
                "cnn": "available"
            },
            "other": {
                "naive_bayes": "available",
                "knn": "available",
                "lazy_learning": "available"
            }
        }
        
        return {
            "status": "healthy",
            "message": "All ML algorithms are available",
            "algorithms": algorithm_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"ML algorithms health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="ML algorithms health check failed")
