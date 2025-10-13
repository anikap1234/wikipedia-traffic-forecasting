"""
K-Nearest Neighbors (KNN) Baseline Model
For Wikipedia Traffic Time Series Forecasting

This implements KNN regression for time series prediction
using weighted average of k-nearest neighbors based on distance metrics.
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time

class KNNTimeSeriesForecaster:
    """
    KNN-based time series forecasting model
    Uses sliding window approach to create training samples
    """
    
    def __init__(self, k=10, window_size=64, distance_metric='canberra', 
                 batch_size=4096):
        """
        Initialize KNN forecaster
        
        Args:
            k: Number of nearest neighbors
            window_size: Size of input window (lookback period)
            distance_metric: Distance metric ('euclidean', 'manhattan', 
                           'canberra', 'minkowski')
            batch_size: Batch size for processing (to manage memory)
        """
        self.k = k
        self.window_size = window_size
        self.distance_metric = distance_metric
        self.batch_size = batch_size
        self.model = None
        self.X_train = None
        self.y_train = None
        
    def create_sequences(self, data, forecast_horizon=1):
        """
        Create input-output pairs using sliding window
        
        Args:
            data: Time series data (n_series, n_timesteps)
            forecast_horizon: Number of steps to forecast
            
        Returns:
            X (input sequences), y (target values)
        """
        n_series, n_timesteps = data.shape
        X, y = [], []
        
        # For each series
        for i in range(n_series):
            series = data[i]
            
            # Create sliding windows
            for t in range(self.window_size, n_timesteps - forecast_horizon + 1):
                X.append(series[t - self.window_size:t])
                y.append(series[t:t + forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def train(self, train_data):
        """
        Train KNN model by storing training samples
        
        Args:
            train_data: Training time series (n_series, n_timesteps)
        """
        print(f"\nTraining KNN (k={self.k}, metric={self.distance_metric})...")
        start_time = time.time()
        
        # Create sequences
        self.X_train, self.y_train = self.create_sequences(train_data, 
                                                           forecast_horizon=1)
        
        print(f"Created {len(self.X_train)} training samples")
        print(f"Input shape: {self.X_train.shape}, Output shape: {self.y_train.shape}")
        
        # Initialize KNN model
        # weights='distance' means closer neighbors have more influence
        self.model = KNeighborsRegressor(
            n_neighbors=self.k,
            weights='distance',  # Inverse distance weighting
            metric=self.distance_metric,
            n_jobs=-1  # Use all available CPU cores
        )
        
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict_single_step(self, input_sequence):
        """
        Predict next value given input sequence
        
        Args:
            input_sequence: Array of shape (window_size,)
            
        Returns:
            Predicted value
        """
        # Reshape for sklearn
        X = input_sequence.reshape(1, -1)
        prediction = self.model.predict(X)
        return prediction[0]
    
    def predict_multi_step(self, initial_sequence, n_steps):
        """
        Predict multiple steps ahead using recursive forecasting
        (predict one step, append to sequence, predict next, etc.)
        
        Args:
            initial_sequence: Initial input sequence (window_size,)
            n_steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(n_steps):
            # Predict next step
            next_val = self.predict_single_step(current_sequence)
            predictions.append(next_val[0])  # Extract scalar
            
            # Update sequence: remove oldest, add newest prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_val[0]
        
        return np.array(predictions)
    
    def predict(self, test_data, forecast_horizon=64):
        """
        Generate predictions for test data
        
        Args:
            test_data: Test time series (n_series, n_timesteps)
            forecast_horizon: Number of steps to forecast for each series
            
        Returns:
            Predictions array (n_series, forecast_horizon)
        """
        print(f"\nGenerating predictions for {test_data.shape[0]} series...")
        print(f"Forecast horizon: {forecast_horizon} steps")
        
        n_series = test_data.shape[0]
        predictions = np.zeros((n_series, forecast_horizon))
        
        # Process in batches to manage memory
        for batch_start in range(0, n_series, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_series)
            batch_size_actual = batch_end - batch_start
            
            print(f"Processing batch {batch_start//self.batch_size + 1}/"
                  f"{(n_series + self.batch_size - 1)//self.batch_size}...", 
                  end='\r')
            
            for i in range(batch_start, batch_end):
                # Use last window_size points as initial sequence
                initial_seq = test_data[i, -self.window_size:]
                
                # Multi-step prediction
                pred = self.predict_multi_step(initial_seq, forecast_horizon)
                predictions[i] = pred
        
        print("\nPrediction completed!")
        return predictions
    
    def evaluate(self, predictions, actual, metric='smape'):
        """
        Evaluate predictions using specified metric
        
        Args:
            predictions: Predicted values
            actual: Actual values
            metric: Evaluation metric ('smape', 'mae', 'rmse')
            
        Returns:
            Metric value
        """
        if metric == 'smape':
            # SMAPE: Symmetric Mean Absolute Percentage Error
            numerator = np.abs(predictions - actual)
            denominator = (np.abs(predictions) + np.abs(actual)) / 2
            # Avoid division by zero
            denominator = np.where(denominator == 0, 1e-8, denominator)
            smape = np.mean(numerator / denominator) * 100
            return smape
        
        elif metric == 'mae':
            return np.mean(np.abs(predictions - actual))
        
        elif metric == 'rmse':
            return np.sqrt(np.mean((predictions - actual) ** 2))
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def visualize_predictions(self, train_data, test_data, predictions, 
                            num_samples=5, save_path='knn_predictions.png'):
        """
        Visualize predictions vs actual values for sample series
        
        Args:
            train_data: Training data for context
            test_data: Test/actual data
            predictions: Predicted values
            num_samples: Number of series to plot
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(predictions))
            
            # Plot training data (last 100 points for context)
            train_series = train_data[idx, -100:]
            train_x = np.arange(len(train_series))
            axes[i].plot(train_x, train_series, 'b-', label='Training', alpha=0.6)
            
            # Plot actual test data
            test_series = test_data[idx, :len(predictions[idx])]
            test_x = np.arange(len(train_series), len(train_series) + len(test_series))
            axes[i].plot(test_x, test_series, 'g-', label='Actual', linewidth=2)
            
            # Plot predictions
            pred_series = predictions[idx]
            axes[i].plot(test_x, pred_series, 'r--', label='KNN Prediction', 
                        linewidth=2)
            
            axes[i].set_title(f'Series {idx} - KNN Forecast')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Normalized Traffic')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.show()


# Hyperparameter tuning helper
def tune_knn_hyperparameters(train_data, val_data, val_actual):
    """
    Tune KNN hyperparameters (k and distance metric)
    
    Args:
        train_data: Training time series
        val_data: Validation input sequences
        val_actual: Validation actual values
        
    Returns:
        Best hyperparameters and their performance
    """
    print("\n" + "="*60)
    print("KNN Hyperparameter Tuning")
    print("="*60)
    
    k_values = [5, 10, 15, 20]
    distance_metrics = ['euclidean', 'manhattan', 'canberra', 'minkowski']
    
    results = []
    
    for k in k_values:
        for metric in distance_metrics:
            print(f"\nTesting k={k}, metric={metric}")
            
            try:
                model = KNNTimeSeriesForecaster(k=k, distance_metric=metric)
                model.train(train_data)
                
                predictions = model.predict(val_data, forecast_horizon=64)
                smape = model.evaluate(predictions, val_actual)
                
                results.append({
                    'k': k,
                    'metric': metric,
                    'smape': smape
                })
                
                print(f"SMAPE: {smape:.2f}")
            
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    # Find best configuration
    best = min(results, key=lambda x: x['smape'])
    print("\n" + "="*60)
    print(f"Best Configuration: k={best['k']}, metric={best['metric']}")
    print(f"Best SMAPE: {best['smape']:.2f}")
    print("="*60)
    
    return best, results


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("KNN Baseline Model - Wikipedia Traffic Forecasting")
    print("="*60)
    
    # Generate synthetic data for demonstration
    n_series = 100
    n_timesteps = 200
    np.random.seed(42)
    
    # Create synthetic time series with trend and seasonality
    t = np.linspace(0, 4*np.pi, n_timesteps)
    synthetic_data = np.zeros((n_series, n_timesteps))
    
    for i in range(n_series):
        trend = 0.01 * t
        seasonal = np.sin(t + np.random.rand() * 2 * np.pi)
        noise = np.random.randn(n_timesteps) * 0.1
        synthetic_data[i] = trend + seasonal + noise
    
    # Split into train and test
    train_data = synthetic_data[:, :150]
    test_data = synthetic_data[:, 150:]
    
    # Initialize and train KNN model
    knn_model = KNNTimeSeriesForecaster(k=10, window_size=32, 
                                        distance_metric='euclidean')
    knn_model.train(train_data)
    
    # Make predictions
    predictions = knn_model.predict(train_data, forecast_horizon=50)
    
    # Evaluate
    smape = knn_model.evaluate(predictions, test_data[:, :50])
    print(f"\nSMAPE: {smape:.2f}")
    
    # Visualize
    knn_model.visualize_predictions(train_data, test_data, predictions, 
                                    num_samples=3)