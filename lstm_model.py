"""
LSTM (Long Short-Term Memory) Recurrent Neural Network
For Wikipedia Traffic Time Series Forecasting

Implements deep LSTM with multiple configurations:
- 10-layer deep LSTM (base model)
- Configurable depth (5, 10, 15 layers)
- Handles long-term temporal dependencies
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import time

class LSTMTimeSeriesForecaster:
    """
    Deep LSTM model for time series forecasting
    Handles multi-layer LSTM with dropout and configurable parameters
    """
    
    def __init__(self, 
                 n_layers=10,
                 hidden_size=80,
                 sequence_length=36,
                 forecast_horizon=64,
                 dropout=0.4,
                 learning_rate=1.0,
                 learning_rate_decay=0.2):
        """
        Initialize LSTM forecaster
        
        Args:
            n_layers: Number of LSTM layers (paper uses 5, 10, or 15)
            hidden_size: Hidden state size (paper: 80, state_size: 45-50)
            sequence_length: Lookback period / backprop length (paper: 30-36)
            forecast_horizon: Number of steps to forecast
            dropout: Dropout rate for regularization (paper: 0.4)
            learning_rate: Initial learning rate (paper: 1.0)
            learning_rate_decay: Final learning rate factor (paper: 0.2)
        """
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        
        self.model = None
        self.history = None
        self.feature_dim = None
        
    def build_model(self, input_dim):
        """
        Build deep LSTM architecture
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        print(f"\nBuilding LSTM model:")
        print(f"  Layers: {self.n_layers}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Forecast horizon: {self.forecast_horizon}")
        
        self.feature_dim = input_dim
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, input_dim))
        
        # Stack LSTM layers
        x = inputs
        for i in range(self.n_layers):
            # return_sequences=True for all but last layer
            return_seq = (i < self.n_layers - 1)
            
            x = layers.LSTM(
                units=self.hidden_size,
                return_sequences=return_seq,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                name=f'lstm_{i+1}'
            )(x)
        
        # Output layer: predict forecast_horizon steps
        outputs = layers.Dense(self.forecast_horizon, name='output')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=self.learning_rate_decay,
            staircase=True
        )
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error
        )
        
        print(f"\nModel architecture:")
        self.model.summary()
        
        return self.model
    
    def create_sequences(self, data, features=None):
        """
        Create input-output sequences for LSTM training
        
        Args:
            data: Time series data (n_series, n_timesteps)
            features: Optional features (n_series, n_timesteps, n_features)
            
        Returns:
            X (inputs), y (targets)
        """
        n_series, n_timesteps = data.shape
        X, y = [], []
        
        for i in range(n_series):
            series = data[i]
            
            # Create sequences
            for t in range(self.sequence_length, 
                          n_timesteps - self.forecast_horizon + 1):
                # Input sequence
                if features is not None:
                    X.append(features[i, t-self.sequence_length:t])
                else:
                    # Use raw values as single feature
                    seq = series[t-self.sequence_length:t]
                    X.append(seq.reshape(-1, 1))
                
                # Target: next forecast_horizon steps
                y.append(series[t:t+self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def train(self, train_data, val_data=None, features_train=None, 
              features_val=None, batch_size=1024, epochs=50, verbose=1):
        """
        Train LSTM model
        
        Args:
            train_data: Training time series (n_series, n_timesteps)
            val_data: Validation time series
            features_train: Training features
            features_val: Validation features
            batch_size: Batch size (paper: 1024 for optimal performance)
            epochs: Number of training epochs
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        print("\n" + "="*60)
        print("Training LSTM Model")
        print("="*60)
        
        # Create sequences
        print("Creating training sequences...")
        X_train, y_train = self.create_sequences(train_data, features_train)
        print(f"Training samples: {len(X_train)}")
        print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
        
        # Build model if not already built
        if self.model is None:
            input_dim = X_train.shape[2]
            self.build_model(input_dim)
        
        # Validation data
        validation_data = None
        if val_data is not None:
            print("Creating validation sequences...")
            X_val, y_val = self.create_sequences(val_data, features_val)
            validation_data = (X_val, y_val)
            print(f"Validation samples: {len(X_val)}")
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if val_data is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_data is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train
        start_time = time.time()
        print(f"\nStarting training (batch_size={batch_size})...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callback_list,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        
        return self.history
    
    def predict(self, test_data, features=None):
        """
        Generate predictions for test data
        
        Args:
            test_data: Test time series (n_series, n_timesteps)
            features: Optional features
            
        Returns:
            Predictions (n_series, forecast_horizon)
        """
        print("\nGenerating predictions...")
        
        n_series = test_data.shape[0]
        predictions = []
        
        for i in range(n_series):
            # Use last sequence_length points
            if features is not None:
                input_seq = features[i, -self.sequence_length:]
                input_seq = input_seq.reshape(1, self.sequence_length, -1)
            else:
                input_seq = test_data[i, -self.sequence_length:]
                input_seq = input_seq.reshape(1, self.sequence_length, 1)
            
            # Predict
            pred = self.model.predict(input_seq, verbose=0)
            predictions.append(pred[0])
        
        return np.array(predictions)
    
    def evaluate_smape(self, predictions, actual):
        """
        Calculate SMAPE metric
        
        Args:
            predictions: Predicted values
            actual: Actual values
            
        Returns:
            SMAPE score
        """
        numerator = np.abs(predictions - actual)
        denominator = (np.abs(predictions) + np.abs(actual)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        smape = np.mean(numerator / denominator) * 100
        return smape
    
    def plot_training_history(self, save_path='lstm_training_history.png'):
        """
        Plot training history (loss curves)
        
        Args:
            save_path: Path to save figure
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.show()
    
    def visualize_predictions(self, train_data, test_data, predictions,
                            num_samples=5, save_path='lstm_predictions.png'):
        """
        Visualize predictions vs actual values
        
        Args:
            train_data: Training data for context
            test_data: Actual test data
            predictions: Predicted values
            num_samples: Number of series to plot
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(predictions))
            
            # Training data (last 100 points)
            train_series = train_data[idx, -100:]
            train_x = np.arange(len(train_series))
            axes[i].plot(train_x, train_series, 'b-', 
                        label='Training', alpha=0.6)
            
            # Actual test data
            test_series = test_data[idx, :len(predictions[idx])]
            test_x = np.arange(len(train_series), 
                             len(train_series) + len(test_series))
            axes[i].plot(test_x, test_series, 'g-', 
                        label='Actual', linewidth=2)
            
            # Predictions
            pred_series = predictions[idx]
            axes[i].plot(test_x, pred_series, 'r--', 
                        label='LSTM Prediction', linewidth=2)
            
            axes[i].set_title(f'Series {idx} - LSTM Forecast')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Normalized Traffic')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.show()
    
    def save_model(self, path='lstm_model.h5'):
        """Save trained model"""
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load_model(self, path='lstm_model.h5'):
        """Load trained model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    print("="*60)
    print("LSTM Model - Wikipedia Traffic Forecasting")
    print("="*60)
    
    # Generate synthetic data
    n_series = 50
    n_timesteps = 300
    np.random.seed(42)
    
    # Create time series with trend and seasonality
    t = np.linspace(0, 6*np.pi, n_timesteps)
    synthetic_data = np.zeros((n_series, n_timesteps))
    
    for i in range(n_series):
        trend = 0.01 * t
        seasonal = np.sin(t + np.random.rand() * 2 * np.pi)
        noise = np.random.randn(n_timesteps) * 0.1
        synthetic_data[i] = trend + seasonal + noise
    
    # Split data
    train_data = synthetic_data[:, :200]
    val_data = synthetic_data[:, 200:250]
    test_data = synthetic_data[:, 250:]
    
    # Initialize LSTM model (10 layers as in paper)
    lstm_model = LSTMTimeSeriesForecaster(
        n_layers=10,
        hidden_size=80,
        sequence_length=36,
        forecast_horizon=50,
        dropout=0.4
    )
    
    # Train
    history = lstm_model.train(
        train_data, 
        val_data=val_data,
        batch_size=64,
        epochs=20
    )
    
    # Plot training history
    lstm_model.plot_training_history()
    
    # Predict
    predictions = lstm_model.predict(val_data)
    
    # Evaluate
    actual = test_data[:, :50]
    smape = lstm_model.evaluate_smape(predictions, actual)
    print(f"\nSMAPE: {smape:.2f}")
    
    # Visualize
    lstm_model.visualize_predictions(train_data, test_data, 
                                     predictions, num_samples=3)