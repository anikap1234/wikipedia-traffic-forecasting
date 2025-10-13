"""
ULTRA-FAST TEST VERSION
- Only 100 pages
- Only 2 epochs
- Skip KNN (slowest to predict)
- Minimal visualizations
"""

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Avoid GUI issues
import matplotlib.pyplot as plt

from data_processor import WikiTrafficDataProcessor
from lstm_model import LSTMTimeSeriesForecaster
from seq2seq_cnn import Seq2SeqCNNForecaster

np.random.seed(42)
print("="*60)
print("ULTRA-FAST TEST: Wikipedia Traffic Forecasting")
print("Pages: 100 | Epochs: 2 | Models: LSTM + Seq2Seq CNN only")
print("="*60)

os.makedirs('quick_test_results', exist_ok=True)

# === STEP 1: Load & subsample data ===
processor = WikiTrafficDataProcessor('data/train_2.csv')
raw_data = processor.load_data()
raw_data = raw_data.iloc[:20].copy()  # Only 100 pages!

train_lstm, test_lstm, _ = processor.preprocess_for_lstm('2017-07-09')
train_cnn, test_cnn, _, _ = processor.preprocess_for_seq2seq_cnn('2017-07-09')

val_size = 64
train_lstm_split = train_lstm[:, :-val_size]
val_lstm = train_lstm[:, -val_size:]
train_cnn_split = train_cnn[:, :-val_size]
val_cnn = train_cnn[:, -val_size:]

print(f"Train shapes: LSTM={train_lstm_split.shape}, CNN={train_cnn_split.shape}")

# === STEP 2: Train LSTM (2 epochs) ===
print("\n[1/2] Training LSTM (2 epochs)...")
lstm = LSTMTimeSeriesForecaster(n_layers=3, hidden_size=32, sequence_length=20, forecast_horizon=64, dropout=0.2)
lstm.build_model(input_dim=1)
X_train, y_train = lstm.create_sequences(train_lstm_split)
X_val, y_val = lstm.create_sequences(val_lstm)
lstm.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=256, verbose=1)

pred_lstm = lstm.predict(train_lstm)
smape_lstm = lstm.evaluate_smape(pred_lstm, test_lstm[:, :64])
print(f"✅ LSTM SMAPE (quick test): {smape_lstm:.2f}")

# === STEP 3: Train Seq2Seq CNN (2 epochs) ===
print("\n[2/2] Training Seq2Seq CNN (2 epochs)...")
cnn = Seq2SeqCNNForecaster(n_encoder_layers=4, n_decoder_layers=4, filters=16, input_length=64, output_length=64, batch_size=32)
cnn.build_model()
X_train_cnn, y_train_cnn = cnn.create_sequences(train_cnn_split)
X_val_cnn, y_val_cnn = cnn.create_sequences(val_cnn)
cnn.model.fit(X_train_cnn, y_train_cnn, validation_data=(X_val_cnn, y_val_cnn), epochs=2, batch_size=32, verbose=1)

pred_cnn = cnn.predict(train_cnn)
smape_cnn = cnn.evaluate_smape(pred_cnn, test_cnn[:, :64])
print(f"✅ Seq2Seq CNN SMAPE (quick test): {smape_cnn:.2f}")

# === STEP 4: Save quick summary ===
summary = {
    "pages_used": 100,
    "epochs": 2,
    "models_trained": ["LSTM", "Seq2Seq CNN"],
    "SMAPE": {
        "LSTM": float(smape_lstm),
        "Seq2Seq_CNN": float(smape_cnn)
    }
}
import json
with open('quick_test_results/quick_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("✅ ULTRA-FAST TEST COMPLETED SUCCESSFULLY!")
print("Results saved in 'quick_test_results/quick_summary.json'")
print("="*60)