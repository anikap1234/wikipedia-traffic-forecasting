"""
DEMO VERSION - 500 Pages, 5 Epochs (Laptop-Friendly)
- Runs in ~60-90 minutes on most laptops
- Uses train_2.csv (correct dataset)
- Includes all 3 models: KNN, LSTM, Seq2Seq CNN
- Generates all visualizations and outputs
- Fixes all known bugs from Knowledge Base
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
print("WIKIPEDIA TRAFFIC FORECASTING - LAPTOP-FRIENDLY DEMO")
print("Using 500 pages, 5 epochs → completes in 60-90 minutes")
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
print("\n✅ Using 500 pages for laptop-friendly demo")
raw_data = raw_data.iloc[:500].copy()
processor.train_data = raw_data  # 🔥 Critical: update processor's data

# Preprocess with sliced data
train_data, test_data, dates = processor.preprocess_for_lstm(train_end_date='2017-07-09')
train_cnn, test_cnn, cond_features, null_indicator = processor.preprocess_for_seq2seq_cnn(train_end_date='2017-07-09')

# Use ALL training data (no validation split for small dataset)
train_split = train_data
val_data = None

train_cnn_split = train_cnn
val_cnn = None

print(f"\nData shapes:")
print(f"  Training: {train_split.shape}")
print(f"  Test: {test_data.shape}")
assert test_data.shape[1] == 64, f"Expected 64 test days, got {test_data.shape[1]}"

# Visualize sample series
processor.visualize_sample_series(num_samples=2)
plt.savefig('demo_results3/data_samples.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# STEP 2a: TRAIN KNN MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 2a: TRAINING KNN BASELINE MODEL")
print("="*70)

knn_model = KNNTimeSeriesForecaster(
    k=10,                    # Paper value (feasible with 500 pages)
    window_size=64,
    distance_metric='canberra',
    batch_size=512           # Reduced from 4096 for RAM
)

knn_model.train(train_split)

# Predict on test set
print("\nGenerating KNN predictions...")
test_predictions_knn = knn_model.predict(train_data, forecast_horizon=64)
test_actual_knn = test_data[:, :64]  # ✅ 64 days (not 32)

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
    save_path='demo_results3/knn_predictions.png'
)
plt.close()

# ============================================================================
# STEP 2b: TRAIN LSTM MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 2b: TRAINING LSTM MODEL (5 epochs)")
print("="*70)

lstm_model = LSTMTimeSeriesForecaster(
    n_layers=10,
    hidden_size=80,
    sequence_length=36,
    forecast_horizon=64,
    dropout=0.4
)

# Train with reduced batch size
history = lstm_model.train(
    train_split,
    val_data=val_data,
    batch_size=256,          # Reduced from 1024
    epochs=5,                # Reduced from 50
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
    save_path='demo_results3/lstm_predictions.png'
)
plt.close()

# ============================================================================
# STEP 2c: TRAIN SEQ2SEQ CNN MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 2c: TRAINING SEQ2SEQ CNN MODEL (5 epochs)")
print("="*70)

cnn_model = Seq2SeqCNNForecaster(
    n_encoder_layers=8,
    n_decoder_layers=8,
    filters=32,
    input_length=128,
    output_length=64,
    batch_size=32,           # Critical for RAM (was 128)
    learning_rate=0.001
)

# Train
history = cnn_model.train(
    train_cnn_split,
    val_data=val_cnn,
    epochs=5,                # Reduced from 50
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
    save_path='demo_results3/seq2seq_cnn_predictions.png'
)
plt.close()

# ============================================================================
# STEP 3: COMPARE RESULTS
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
print("MODEL PERFORMANCE COMPARISON (500 pages, 5 epochs)")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

# Save results
results_df.to_csv('demo_results3/model_comparison.csv', index=False)

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
plt.title('Model Performance Comparison - Laptop Demo\n(500 pages, 5 epochs)', 
         fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('demo_results3/model_comparison.png', dpi=150, bbox_inches='tight')
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
plt.savefig('demo_results3/all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Save summary (with Python float conversion)
summary = {
    'demo_settings': {
        'n_pages': 500,
        'epochs': 5,
        'note': 'Laptop-friendly demo with all models'
    },
    'results': {
        'KNN': float(knn_smape),
        'LSTM': float(lstm_smape),
        'Seq2Seq_CNN': float(cnn_smape)
    },
    'best_model': results_df.iloc[0]['Model'],
    'improvement_vs_baseline': f"{((knn_smape - cnn_smape) / knn_smape * 100):.1f}%"
}

with open('demo_results3/results_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print(f"\n✓ All results saved to 'demo_results/' directory")
print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "="*70)
print("LAPTOP-FRIENDLY DEMO COMPLETED SUCCESSFULLY!")
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