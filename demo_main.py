"""
ULTRA-STABLE DEMO VERSION - FIXED FOR train_2.csv
- Uses train_2.csv (correct dataset)
- Fixes data slicing bug
- Handles 10 pages + 2 epochs safely
- Generates all visualizations and outputs
- No more shape errors or memory issues
"""

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Fix for headless environments (Colab, servers, Windows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Import model classes
from data_processor import WikiTrafficDataProcessor
from knn_model import KNNTimeSeriesForecaster
from lstm_model import LSTMTimeSeriesForecaster
from seq2seq_cnn import Seq2SeqCNNForecaster

print("="*70)
print("WIKIPEDIA TRAFFIC FORECASTING - ULTRA-FAST DEMO")
print("Using 10 pages, 2 epochs → completes in 2-3 minutes")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Create results directory
os.makedirs('demo_results', exist_ok=True)

# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA (CORRECTED)
# ============================================================================
print("\n" + "="*70)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("="*70)

# ✅ USE train_2.csv (NOT train_1.csv)
processor = WikiTrafficDataProcessor('data/train_2.csv')
raw_data = processor.load_data()

# ✅ SLICE BEFORE preprocessing + update processor's internal data
print("\n✅ Using only 10 pages for ultra-fast demo")
raw_data = raw_data.iloc[:10].copy()
processor.train_data = raw_data  # 🔥 CRITICAL: update processor's data

# Preprocess with sliced data
train_data, test_data, dates = processor.preprocess_for_lstm(train_end_date='2017-07-09')
train_cnn, test_cnn, cond_features, null_indicator = processor.preprocess_for_seq2seq_cnn(train_end_date='2017-07-09')

# ✅ VALIDATE data shapes
print(f"Train data shape: {train_data.shape}")  # Should be (10, 740)
print(f"Test data shape: {test_data.shape}")    # Should be (10, 64)
assert train_data.shape[1] >= 100, f"Need at least 100 training timesteps, got {train_data.shape[1]}"
assert test_data.shape[1] == 64, f"Expected 64 test days, got {test_data.shape[1]}"

# Use ALL training data (no validation split for tiny dataset)
train_split = train_data  # Keep all ~740 timesteps
val_data = None  # No validation needed for 10 pages

train_cnn_split = train_cnn
val_cnn = None

print(f"\nFinal data shapes:")
print(f"  Training: {train_split.shape}")
print(f"  Test: {test_data.shape}")

# Visualize sample series
processor.visualize_sample_series(num_samples=2)
plt.savefig('demo_results/data_samples.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# STEP 2: TRAIN KNN MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 2a: TRAINING KNN BASELINE MODEL")
print("="*70)

knn_model = KNNTimeSeriesForecaster(
    k=2,
    window_size=32,        # Reduced window for small data
    distance_metric='canberra',
    batch_size=5           # Very small batch
)

knn_model.train(train_split)

# Predict on test set
print("\nGenerating KNN predictions...")
test_predictions_knn = knn_model.predict(train_data, forecast_horizon=64)
test_actual_knn = test_data[:, :64]  # ✅ 64 days

# Calculate SMAPE
def calculate_smape(predictions, actual):
    numerator = np.abs(predictions - actual)
    denominator = (np.abs(predictions) + np.abs(actual)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.mean(numerator / denominator) * 100
    return smape

knn_smape = calculate_smape(test_predictions_knn, test_actual_knn)
print(f"\n✓ KNN Test SMAPE: {knn_smape:.2f}")

# Visualize
knn_model.visualize_predictions(
    train_data, test_data, test_predictions_knn,
    num_samples=2,
    save_path='demo_results/knn_predictions.png'
)
plt.close()

# ============================================================================
# STEP 3: TRAIN LSTM MODEL (FIXED)
# ============================================================================
print("\n" + "="*70)
print("STEP 2b: TRAINING LSTM MODEL (2 epochs)")
print("="*70)

lstm_model = LSTMTimeSeriesForecaster(
    n_layers=3,            # Reduced layers
    hidden_size=32,        # Reduced size
    sequence_length=36,
    forecast_horizon=64,
    dropout=0.2            # Reduced dropout
)

# Train with NO validation (val_data=None)
history = lstm_model.train(
    train_split,
    val_data=None,         # 🔥 KEY FIX: No validation for tiny data
    batch_size=8,          # Small batch to avoid memory issues
    epochs=2,              # Only 2 epochs
    verbose=1
)

# Plot training history
lstm_model.plot_training_history(save_path='demo_results/lstm_training_history.png')
plt.close()

# Test predictions
print("\nGenerating LSTM predictions...")
test_predictions_lstm = lstm_model.predict(train_data)
test_actual_lstm = test_data[:, :64]
lstm_smape = calculate_smape(test_predictions_lstm, test_actual_lstm)
print(f"\n✓ LSTM Test SMAPE: {lstm_smape:.2f}")

# Visualize
lstm_model.visualize_predictions(
    train_data, test_data, test_predictions_lstm,
    num_samples=2,
    save_path='demo_results/lstm_predictions.png'
)
plt.close()

# ============================================================================
# STEP 4: TRAIN SEQ2SEQ CNN MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 2c: TRAINING SEQ2SEQ CNN MODEL (2 epochs)")
print("="*70)

# Ensure enough timesteps for CNN
assert train_cnn_split.shape[1] >= 128 + 64, f"CNN needs 192 timesteps, got {train_cnn_split.shape[1]}"

# SEQ2SEQ CNN MODEL (WORKING CONFIGURATION)
cnn_model = Seq2SeqCNNForecaster(
    n_encoder_layers=4,    
    n_decoder_layers=4,    
    filters=32,            # Keep at 32 to match ResidualBlock defaults
    input_length=128,      
    output_length=64,
    batch_size=4,
    learning_rate=0.001
)

# Train with NO validation
history = cnn_model.train(
    train_cnn_split,
    val_data=None,         # No validation
    epochs=2,              # Only 2 epochs
    verbose=1
)

# Test predictions
print("\nGenerating Seq2Seq CNN predictions...")
test_predictions_cnn = cnn_model.predict(train_cnn)
test_actual_cnn = test_cnn[:, :64]
cnn_smape = calculate_smape(test_predictions_cnn, test_actual_cnn)
print(f"\n✓ Seq2Seq CNN Test SMAPE: {cnn_smape:.2f}")

# Visualize
cnn_model.visualize_predictions(
    train_cnn, test_cnn, test_predictions_cnn,
    num_samples=2,
    save_path='demo_results/seq2seq_cnn_predictions.png'
)
plt.close()

# ============================================================================
# STEP 5: COMPARE RESULTS
# ============================================================================
print("\n" + "="*70)
print("STEP 3: FINAL RESULTS COMPARISON")
print("="*70)

# Create comparison table
results = {
    'Model': ['KNN', 'LSTM', 'Seq2Seq CNN'],
    'SMAPE': [knn_smape, lstm_smape, cnn_smape]
}

results_df = pd.DataFrame(results).sort_values('SMAPE')

print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON (Ultra-Fast Demo - 10 pages)")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

# Save results
results_df.to_csv('demo_results/model_comparison.csv', index=False)

# Create bar plot
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = plt.bar(results_df['Model'], results_df['SMAPE'], color=colors)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('SMAPE (%)', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison - Ultra-Fast Demo\n(10 pages, 2 epochs)', 
         fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('demo_results/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Create side-by-side comparison (1 row for speed)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

idx = np.random.randint(0, len(test_predictions_knn))
for j, (model_name, predictions, color) in enumerate([
    ('KNN', test_predictions_knn, '#FF6B6B'),
    ('LSTM', test_predictions_lstm, '#4ECDC4'),
    ('Seq2Seq CNN', test_predictions_cnn, '#45B7D1')
]):
    ax = axes[j]
    
    # Training data (last 100 points)
    train_series = train_data[idx, -100:]
    train_x = np.arange(len(train_series))
    ax.plot(train_x, train_series, 'gray', label='Training', alpha=0.5, linewidth=1)
    
    # Actual test
    test_series = test_data[idx, :64]
    test_x = np.arange(len(train_series), len(train_series) + len(test_series))
    ax.plot(test_x, test_series, 'g-', label='Actual', linewidth=2.5)
    
    # Predictions
    pred_series = predictions[idx]
    ax.plot(test_x, pred_series, '--', color=color, 
           label=f'{model_name}', linewidth=2)
    
    # Calculate series SMAPE
    series_smape = calculate_smape(pred_series[:len(test_series)], test_series)
    
    ax.set_title(f'{model_name} - Series {idx}\nSMAPE: {series_smape:.2f}%',
               fontsize=10, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=9)
    ax.set_ylabel('Normalized Traffic', fontsize=9)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('demo_results/all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Save summary (with Python float conversion)
summary = {
    'demo_settings': {
        'n_pages': 10,
        'epochs': 2,
        'note': 'Ultra-fast demo for testing'
    },
    'results': {
        'KNN': float(knn_smape),
        'LSTM': float(lstm_smape),
        'Seq2Seq_CNN': float(cnn_smape)
    },
    'best_model': results_df.iloc[0]['Model'],
    'improvement_vs_baseline': f"{((knn_smape - cnn_smape) / knn_smape * 100):.1f}%"
}

with open('demo_results/results_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print(f"\n✓ All results saved to 'demo_results/' directory")
print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "="*70)
print("ULTRA-FAST DEMO COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated files:")
print("  📊 demo_results/model_comparison.png")
print("  📊 demo_results/all_models_comparison.png")
print("  📊 demo_results/knn_predictions.png")
print("  📊 demo_results/lstm_predictions.png")
print("  📊 demo_results/seq2seq_cnn_predictions.png")
print("  📊 demo_results/data_samples.png")
print("  📊 demo_results/lstm_training_history.png")
print("  📄 demo_results/results_summary.json")
print("  📄 demo_results/model_comparison.csv")