"""
Wikipedia Web Traffic Time Series Forecasting
UE23CS352A ML Project Implementation

This module implements three approaches for time series forecasting:
1. KNN (Baseline)
2. LSTM Recurrent Network
3. Sequence-to-Sequence with Causal CNN

Dataset: Wikipedia page view counts (July 2015 - September 2017)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class WikiTrafficDataProcessor:
    """
    Handles data loading, preprocessing, and feature extraction
    for Wikipedia traffic forecasting
    """
    
    def __init__(self, train_file_path):
        """
        Initialize the data processor
        
        Args:
            train_file_path: Path to training CSV file (train_*.csv)
        """
        self.train_file = train_file_path
        self.train_data = None
        self.processed_data = None
        self.feature_data = None
        
    def load_data(self):
        """
        Load Wikipedia traffic data from CSV
        
        Returns:
            DataFrame with pages as rows and dates as columns
        """
        print("Loading data...")
        self.train_data = pd.read_csv(self.train_file)
        print(f"Data loaded: {self.train_data.shape[0]} pages, {self.train_data.shape[1]-1} days")
        return self.train_data
    
    def parse_page_info(self, page_name):
        """
        Extract metadata from page name
        Format: 'name_project_access_agent'
        
        Args:
            page_name: Full Wikipedia page name
            
        Returns:
            Dictionary with parsed components
        """
        parts = page_name.rsplit('_', 3)
        
        if len(parts) == 4:
            name, project, access, agent = parts
        else:
            # Handle edge cases
            name = '_'.join(parts[:-3]) if len(parts) > 3 else parts[0]
            project = parts[-3] if len(parts) > 3 else 'unknown'
            access = parts[-2] if len(parts) > 2 else 'unknown'
            agent = parts[-1] if len(parts) > 1 else 'unknown'
        
        # Extract country code from project
        country = project.split('.')[0] if '.' in project else 'unknown'
        
        return {
            'name': name,
            'project': project,
            'access': access,
            'agent': agent,
            'country': country
        }
    
    def preprocess_for_lstm(self, train_end_date='2017-07-09'):
        """
        Preprocess data for LSTM approach:
        1. Handle missing values (NaN)
        2. Apply log(x+1) transformation
        3. Normalize to zero mean and unit variance
        
        Args:
            train_end_date: Split date for train/test
            
        Returns:
            Tuple of (normalized_train, normalized_test, metadata)
        """
        print("\nPreprocessing for LSTM...")
        
        # Get date columns (all except 'Page')
        date_cols = [col for col in self.train_data.columns if col != 'Page']
        
        # Convert to datetime for easier manipulation
        date_cols_dt = pd.to_datetime(date_cols)
        split_idx = date_cols_dt.get_loc(pd.to_datetime(train_end_date))
        
        # Extract time series values
        values = self.train_data[date_cols].values
        
        # Step 1: Handle missing values
        # Missing values are already NaN in the dataset
        # We'll fill NaN with 0 for now (as mentioned in paper)
        values_filled = np.nan_to_num(values, nan=0.0)
        
        # Step 2: Apply log(x+1) transformation to handle zeros and large spikes
        values_log = np.log1p(values_filled)  # log1p = log(x+1)
        
        # Step 3: Normalize each time series to zero mean and unit variance
        normalized_values = np.zeros_like(values_log)
        for i in range(values_log.shape[0]):
            series = values_log[i]
            mean = np.mean(series)
            std = np.std(series)
            if std > 0:
                normalized_values[i] = (series - mean) / std
            else:
                normalized_values[i] = series - mean
        
        # Split into train and validation/test
        train_values = normalized_values[:, :split_idx]
        test_values = normalized_values[:, split_idx:]
        
        print(f"Train shape: {train_values.shape}, Test shape: {test_values.shape}")
        
        return train_values, test_values, date_cols_dt
    
    def compute_fft_periodicity(self, series, sample_rate=1.0):
        """
        Compute FFT to identify periodicity in time series
        
        Args:
            series: Time series array
            sample_rate: Sampling rate (1.0 for daily data)
            
        Returns:
            Frequencies and magnitudes
        """
        # Remove mean
        series_detrended = series - np.mean(series)
        
        # Compute FFT
        n = len(series_detrended)
        fft_vals = fft(series_detrended)
        fft_freq = fftfreq(n, d=sample_rate)
        
        # Only positive frequencies
        pos_mask = fft_freq > 0
        freqs = fft_freq[pos_mask]
        magnitudes = np.abs(fft_vals[pos_mask])
        
        return freqs, magnitudes
    
    def extract_features_for_lstm(self, series, date_index):
        """
        Extract features for LSTM model:
        - log(hit count + 1)
        - day of week (one-hot encoded, normalized)
        - monthly autocorrelation
        - quarterly autocorrelation  
        - annual autocorrelation
        - traffic median
        
        Args:
            series: Time series array
            date_index: DatetimeIndex for the series
            
        Returns:
            Feature array
        """
        features = []
        
        for i in range(len(series)):
            feat = []
            
            # 1. Current value (already log-normalized)
            feat.append(series[i])
            
            # 2. Day of week (one-hot encoded)
            dow = date_index[i].dayofweek  # 0=Monday, 6=Sunday
            dow_onehot = np.zeros(7)
            dow_onehot[dow] = 1
            # Normalize to zero mean and unit variance
            dow_onehot = (dow_onehot - np.mean(dow_onehot)) / (np.std(dow_onehot) + 1e-8)
            feat.extend(dow_onehot)
            
            # 3. Autocorrelation features (if enough history)
            # Monthly (30 days lag)
            if i >= 30:
                monthly_autocorr = np.corrcoef(series[i-30:i], series[i-29:i+1])[0, 1]
                feat.append(monthly_autocorr if not np.isnan(monthly_autocorr) else 0)
            else:
                feat.append(0)
            
            # Quarterly (90 days lag)
            if i >= 90:
                quarterly_autocorr = np.corrcoef(series[i-90:i], series[i-89:i+1])[0, 1]
                feat.append(quarterly_autocorr if not np.isnan(quarterly_autocorr) else 0)
            else:
                feat.append(0)
            
            # Annual (365 days lag)
            if i >= 365:
                annual_autocorr = np.corrcoef(series[i-365:i], series[i-364:i+1])[0, 1]
                feat.append(annual_autocorr if not np.isnan(annual_autocorr) else 0)
            else:
                feat.append(0)
            
            # 4. Traffic median (rolling window of last 30 days)
            if i >= 30:
                feat.append(np.median(series[i-30:i]))
            else:
                feat.append(np.median(series[:i+1]) if i > 0 else series[i])
            
            features.append(feat)
        
        return np.array(features)
    
    def preprocess_for_seq2seq_cnn(self, train_end_date='2017-07-09'):
        """
        Preprocess data for Seq2Seq CNN approach:
        1. Apply log transformation
        2. Normalize to zero mean
        3. Create conditional features (project, access, agent, null indicator, log mean)
        
        Args:
            train_end_date: Split date for train/test
            
        Returns:
            Tuple of processed data and conditional features
        """
        print("\nPreprocessing for Seq2Seq CNN...")
        
        # Get date columns
        date_cols = [col for col in self.train_data.columns if col != 'Page']
        date_cols_dt = pd.to_datetime(date_cols)
        split_idx = date_cols_dt.get_loc(pd.to_datetime(train_end_date))
        
        # Extract values
        values = self.train_data[date_cols].values
        
        # Track null values before filling
        null_indicator = np.isnan(values).astype(float)
        
        # Fill NaN with 0
        values_filled = np.nan_to_num(values, nan=0.0)
        
        # Apply log transformation
        values_log = np.log1p(values_filled)
        
        # Normalize to zero mean (per series)
        normalized_values = values_log - np.mean(values_log, axis=1, keepdims=True)
        
        # Calculate log mean for each series
        log_means = np.mean(values_log, axis=1)
        
        # Parse page metadata for conditional features
        page_features = []
        for page in self.train_data['Page']:
            info = self.parse_page_info(page)
            page_features.append(info)
        
        page_df = pd.DataFrame(page_features)
        
        # One-hot encode categorical features
        project_encoded = pd.get_dummies(page_df['project'], prefix='project')
        access_encoded = pd.get_dummies(page_df['access'], prefix='access')
        agent_encoded = pd.get_dummies(page_df['agent'], prefix='agent')
        
        # Combine all conditional features
        conditional_features = pd.concat([
            project_encoded,
            access_encoded,
            agent_encoded
        ], axis=1)
        
        # Add log mean and null indicator ratio
        conditional_features['log_mean'] = log_means
        conditional_features['null_ratio'] = null_indicator.mean(axis=1)
        
        # Split data
        train_values = normalized_values[:, :split_idx]
        test_values = normalized_values[:, split_idx:]
        
        print(f"Train shape: {train_values.shape}")
        print(f"Conditional features shape: {conditional_features.shape}")
        
        return train_values, test_values, conditional_features.values, null_indicator
    
    def visualize_sample_series(self, num_samples=3):
        """
        Visualize sample time series with normalization and FFT
        
        Args:
            num_samples: Number of random series to plot
        """
        date_cols = [col for col in self.train_data.columns if col != 'Page']
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4*num_samples))
        
        for i in range(num_samples):
            # Random series
            idx = np.random.randint(0, len(self.train_data))
            series = self.train_data.iloc[idx][date_cols].values
            series = np.array(series, dtype=np.float64)
            series_filled = np.nan_to_num(series, nan=0.0)
            series_log = np.log1p(series_filled)
            
            # Plot normalized series
            axes[i, 0].plot(series_log)
            axes[i, 0].set_title(f'Log(x+1) Normalized Series {i+1}')
            axes[i, 0].set_xlabel('Days')
            axes[i, 0].set_ylabel('Log(Views + 1)')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Compute and plot FFT
            freqs, magnitudes = self.compute_fft_periodicity(series_log)
            
            # Convert frequency to period (days)
            periods = 1.0 / (freqs + 1e-10)
            
            axes[i, 1].plot(periods, magnitudes)
            axes[i, 1].set_title(f'FFT - Periodicity Analysis {i+1}')
            axes[i, 1].set_xlabel('Period (days)')
            axes[i, 1].set_ylabel('Magnitude')
            axes[i, 1].set_xlim(0, 30)  # Focus on periods up to 30 days
            axes[i, 1].axvline(x=7, color='r', linestyle='--', label='Weekly')
            axes[i, 1].axvline(x=3, color='g', linestyle='--', label='3-day')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_visualization.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved as 'data_visualization.png'")
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Wikipedia Traffic Forecasting - Data Preprocessing")
    print("=" * 60)
    
    # Initialize processor
    # Replace with your actual file path
    processor = WikiTrafficDataProcessor('..data/train_2.csv')
    
    # Load data
    data = processor.load_data()
    
    # Preprocess for LSTM
    train_lstm, test_lstm, dates = processor.preprocess_for_lstm()
    
    # Preprocess for Seq2Seq CNN
    train_cnn, test_cnn, cond_feat, null_ind = processor.preprocess_for_seq2seq_cnn()
    
    # Visualize sample series
    processor.visualize_sample_series(num_samples=3)
    
    print("\n" + "=" * 60)
    print("Data preprocessing completed successfully!")
    print("=" * 60)