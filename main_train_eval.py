"""
Main Training and Evaluation Script
Wikipedia Traffic Time Series Forecasting Project

Trains all three models (KNN, LSTM, Seq2Seq CNN) and compares results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# Import our model classes (assuming they're in the same directory)
# from data_processor import WikiTrafficDataProcessor
# from knn_model import KNNTimeSeriesForecaster
# from lstm_model import LSTMTimeSeriesForecaster
# from seq2seq_cnn import Seq2SeqCNNForecaster


def calculate_smape(predictions, actual):
    """
    Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
    
    Formula: SMAPE = (1/n) * Σ |F_t - A_t| / ((|F_t| + |A_t|) / 2)
    
    Args:
        predictions: Predicted values
        actual: Actual values
        
    Returns:
        SMAPE percentage
    """
    numerator = np.abs(predictions - actual)
    denominator = (np.abs(predictions) + np.abs(actual)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(numerator / denominator) * 100
    return smape


class ExperimentRunner:
    """
    Manages training and evaluation of all models
    """
    
    def __init__(self, data_path, results_dir='results'):
        """
        Initialize experiment runner
        
        Args:
            data_path: Path to training data CSV
            results_dir: Directory to save results
        """
        self.data_path = data_path
        self.results_dir = results_dir
        self.processor = None
        self.results = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess data for all models
        """
        print("\n" + "="*70)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*70)
        
        # Initialize processor
        from data_processor import WikiTrafficDataProcessor
        self.processor = WikiTrafficDataProcessor(self.data_path)
        
        # Load data
        self.raw_data = self.processor.load_data()
        
        # Preprocess for LSTM
        print("\nPreprocessing for LSTM...")
        self.train_lstm, self.test_lstm, self.dates = \
            self.processor.preprocess_for_lstm(train_end_date='2017-07-09')
        
        # Preprocess for Seq2Seq CNN
        print("\nPreprocessing for Seq2Seq CNN...")
        self.train_cnn, self.test_cnn, self.cond_features, self.null_indicator = \
            self.processor.preprocess_for_seq2seq_cnn(train_end_date='2017-07-09')
        
        # For KNN, use LSTM preprocessing
        self.train_knn = self.train_lstm
        self.test_knn = self.test_lstm
        
        # Split validation set (last 64 days of training)
        self.val_size = 64
        
        self.train_lstm_split = self.train_lstm[:, :-self.val_size]
        self.val_lstm = self.train_lstm[:, -self.val_size:]
        
        self.train_cnn_split = self.train_cnn[:, :-self.val_size]
        self.val_cnn = self.train_cnn[:, -self.val_size:]
        
        self.train_knn_split = self.train_knn[:, :-self.val_size]
        self.val_knn = self.train_knn[:, -self.val_size:]
        
        print("\nData preprocessing completed!")
        print(f"Train shape (LSTM): {self.train_lstm_split.shape}")
        print(f"Validation shape: {self.val_lstm.shape}")
        print(f"Test shape: {self.test_lstm.shape}")
        
        # Visualize sample series
        self.processor.visualize_sample_series(num_samples=3)
    
    def train_knn(self):
        """
        Train KNN baseline model
        """
        print("\n" + "="*70)
        print("STEP 2a: TRAINING KNN BASELINE MODEL")
        print("="*70)
        
        from knn_model import KNNTimeSeriesForecaster
        
        # Initialize KNN with best hyperparameters from paper
        self.knn_model = KNNTimeSeriesForecaster(
            k=10,
            window_size=64,
            distance_metric='canberra',
            batch_size=4096
        )
        
        # Train
        self.knn_model.train(self.train_knn_split)
        
        # Validate
        print("\nValidating KNN model...")
        val_predictions_knn = self.knn_model.predict(
            self.train_knn_split, 
            forecast_horizon=64
        )
        val_smape_knn = calculate_smape(val_predictions_knn, self.val_knn)
        print(f"Validation SMAPE: {val_smape_knn:.2f}")
        
        # Test
        print("\nTesting KNN model...")
        test_predictions_knn = self.knn_model.predict(
            self.train_knn,
            forecast_horizon=64
        )
        test_actual_knn = self.test_knn[:, :64]
        test_smape_knn = calculate_smape(test_predictions_knn, test_actual_knn)
        print(f"Test SMAPE: {test_smape_knn:.2f}")
        
        # Save results
        self.results['KNN'] = {
            'val_smape': float(val_smape_knn),
            'test_smape': float(test_smape_knn),
            'predictions': test_predictions_knn
        }
        
        # Visualize
        self.knn_model.visualize_predictions(
            self.train_knn, 
            self.test_knn, 
            test_predictions_knn,
            num_samples=5,
            save_path=os.path.join(self.results_dir, 'knn_predictions.png')
        )
    
    def train_lstm(self, epochs=50):
        """
        Train LSTM model
        """
        print("\n" + "="*70)
        print("STEP 2b: TRAINING LSTM MODEL")
        print("="*70)
        
        from lstm_model import LSTMTimeSeriesForecaster
        
        # Initialize LSTM with paper's hyperparameters
        self.lstm_model = LSTMTimeSeriesForecaster(
            n_layers=10,
            hidden_size=80,
            sequence_length=36,
            forecast_horizon=64,
            dropout=0.4,
            learning_rate=1.0,
            learning_rate_decay=0.2
        )
        
        # Train
        history = self.lstm_model.train(
            self.train_lstm_split,
            val_data=self.val_lstm,
            batch_size=1024,
            epochs=epochs,
            verbose=1
        )
        
        # Plot training history
        self.lstm_model.plot_training_history(
            save_path=os.path.join(self.results_dir, 'lstm_training_history.png')
        )
        
        # Test
        print("\nTesting LSTM model...")
        test_predictions_lstm = self.lstm_model.predict(self.train_lstm)
        test_actual_lstm = self.test_lstm[:, :64]
        test_smape_lstm = calculate_smape(test_predictions_lstm, test_actual_lstm)
        print(f"Test SMAPE: {test_smape_lstm:.2f}")
        
        # Save results
        self.results['LSTM'] = {
            'test_smape': float(test_smape_lstm),
            'predictions': test_predictions_lstm,
            'history': history.history
        }
        
        # Visualize
        self.lstm_model.visualize_predictions(
            self.train_lstm,
            self.test_lstm,
            test_predictions_lstm,
            num_samples=5,
            save_path=os.path.join(self.results_dir, 'lstm_predictions.png')
        )
        
        # Save model
        self.lstm_model.save_model(
            os.path.join(self.results_dir, 'lstm_model.h5')
        )
    
    def train_seq2seq_cnn(self, epochs=50):
        """
        Train Seq2Seq CNN model
        """
        print("\n" + "="*70)
        print("STEP 2c: TRAINING SEQ2SEQ CNN MODEL")
        print("="*70)
        
        from seq2seq_cnn import Seq2SeqCNNForecaster
        
        # Initialize Seq2Seq CNN with paper's hyperparameters
        self.cnn_model = Seq2SeqCNNForecaster(
            n_encoder_layers=8,
            n_decoder_layers=8,
            filters=32,
            input_length=128,
            output_length=64,
            batch_size=128,
            learning_rate=0.001
        )
        
        # Train
        history = self.cnn_model.train(
            self.train_cnn_split,
            val_data=self.val_cnn,
            epochs=epochs,
            verbose=1
        )
        
        # Test
        print("\nTesting Seq2Seq CNN model...")
        test_predictions_cnn = self.cnn_model.predict(self.train_cnn)
        test_actual_cnn = self.test_cnn[:, :64]
        test_smape_cnn = calculate_smape(test_predictions_cnn, test_actual_cnn)
        print(f"Test SMAPE: {test_smape_cnn:.2f}")
        
        # Save results
        self.results['Seq2Seq_CNN'] = {
            'test_smape': float(test_smape_cnn),
            'predictions': test_predictions_cnn,
            'history': history.history
        }
        
        # Visualize
        self.cnn_model.visualize_predictions(
            self.train_cnn,
            self.test_cnn,
            test_predictions_cnn,
            num_samples=5,
            save_path=os.path.join(self.results_dir, 'seq2seq_cnn_predictions.png')
        )
    
    def compare_results(self):
        """
        Compare all models and create summary visualizations
        """
        print("\n" + "="*70)
        print("STEP 3: RESULTS COMPARISON")
        print("="*70)
        
        # Create comparison table
        comparison_data = {
            'Model': [],
            'SMAPE': []
        }
        
        for model_name, results in self.results.items():
            comparison_data['Model'].append(model_name)
            comparison_data['SMAPE'].append(results['test_smape'])
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('SMAPE')
        
        print("\n" + "="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print("="*70)
        
        # Save comparison table
        comparison_df.to_csv(
            os.path.join(self.results_dir, 'model_comparison.csv'),
            index=False
        )
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(comparison_df['Model'], comparison_df['SMAPE'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.xlabel('Model', fontsize=12, fontweight='bold')
        plt.ylabel('SMAPE (%)', fontsize=12, fontweight='bold')
        plt.title('Model Performance Comparison\n(Lower is Better)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, 'model_comparison.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.show()
        
        # Create side-by-side prediction comparison for sample series
        self.create_comparison_plot()
        
        # Save all results to JSON
        results_to_save = {}
        for model_name, results in self.results.items():
            results_to_save[model_name] = {
                'test_smape': results['test_smape']
            }
        
        with open(os.path.join(self.results_dir, 'results_summary.json'), 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        print(f"\nAll results saved to '{self.results_dir}' directory")
    
    def create_comparison_plot(self, num_samples=3):
        """
        Create side-by-side comparison of all models' predictions
        """
        fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4*num_samples))
        
        model_names = ['KNN', 'LSTM', 'Seq2Seq_CNN']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(self.results['KNN']['predictions']))
            
            for j, model_name in enumerate(model_names):
                ax = axes[i, j] if num_samples > 1 else axes[j]
                
                # Training data (last 100 points)
                train_series = self.train_lstm[idx, -100:]
                train_x = np.arange(len(train_series))
                ax.plot(train_x, train_series, 'gray', 
                       label='Training', alpha=0.5, linewidth=1)
                
                # Actual test data
                test_series = self.test_lstm[idx, :64]
                test_x = np.arange(len(train_series), 
                                 len(train_series) + len(test_series))
                ax.plot(test_x, test_series, 'g-', 
                       label='Actual', linewidth=2.5)
                
                # Predictions
                pred_series = self.results[model_name]['predictions'][idx]
                ax.plot(test_x, pred_series, '--', 
                       color=colors[j],
                       label=f'{model_name} Prediction', 
                       linewidth=2)
                
                # Calculate SMAPE for this series
                series_smape = calculate_smape(
                    pred_series[:len(test_series)], 
                    test_series
                )
                
                ax.set_title(f'{model_name} - Series {idx}\nSMAPE: {series_smape:.2f}%',
                           fontsize=11, fontweight='bold')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Normalized Traffic')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, 'all_models_comparison.png'),
            dpi=150, bbox_inches='tight'
        )
        print(f"\nComparison plot saved!")
        plt.show()
    
    def run_full_experiment(self, train_lstm_epochs=50, train_cnn_epochs=50):
        """
        Run complete experiment pipeline
        """
        print("\n" + "="*70)
        print("WIKIPEDIA TRAFFIC FORECASTING - FULL EXPERIMENT")
        print("="*70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Train all models
        self.train_knn()
        self.train_lstm(epochs=train_lstm_epochs)
        self.train_seq2seq_cnn(epochs=train_cnn_epochs)
        
        # Step 3: Compare results
        self.compare_results()
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)


if __name__ == "__main__":
    # Configuration
    DATA_PATH = 'train_2.csv'  # Update with your data path
    RESULTS_DIR = 'results'
    
    # Run experiment
    runner = ExperimentRunner(DATA_PATH, RESULTS_DIR)
    
    # For quick testing, use fewer epochs
    # For full experiment as in paper, use more epochs (50+)
    runner.run_full_experiment(
        train_lstm_epochs=20,  # Increase for better results
        train_cnn_epochs=20    # Increase for better results
    )