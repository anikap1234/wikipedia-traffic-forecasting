# Wikipedia Web Traffic Time Series Forecasting

**Course**: UE23CS352A - Machine Learning  
**Project**: Time Series Forecasting using Machine Learning

## 📋 Project Overview

This project implements and compares three machine learning approaches for forecasting Wikipedia web traffic time series data:

1. **K-Nearest Neighbors (KNN)** - Baseline model
2. **LSTM (Long Short-Term Memory)** - Deep recurrent neural network
3. **Sequence-to-Sequence with Causal CNN** - Convolutional encoder-decoder architecture

The dataset contains approximately 145,000 time series of daily Wikipedia page views from July 2015 to September 2017.

## 🎯 Problem Statement

Accurate time series forecasting is critical for:
- Resource allocation and capacity planning
- Budget optimization
- Anomaly detection in web traffic
- Understanding trends and seasonality

**Challenge**: Web traffic data is noisy, has missing values, exhibits outlier spikes, and shows variable patterns across different pages.

## 📊 Dataset

**Source**: [Kaggle - Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting)

**Details**:
- ~145,000 time series
- Daily page views from July 1, 2015 to September 10, 2017
- Training/validation split: July 9, 2017
- Each series has metadata: Wikipedia project, access type, agent type

**Features**:
- Page name (contains project, access type, agent)
- Time series values (daily page views)
- Missing values represented as NaN

## 🏗️ Project Structure

```
wikipedia-traffic-forecasting/
│
├── data_processor.py          # Data loading and preprocessing
├── knn_model.py               # KNN baseline implementation
├── lstm_model.py              # LSTM model implementation
├── seq2seq_cnn.py             # Seq2Seq CNN implementation
├── main_train_eval.py         # Main training & evaluation script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/                      # Data directory (create this)
│   └── train_2.csv           # Training data (download from Kaggle)
│
├── results/                   # Results directory (auto-created)
│   ├── knn_predictions.png
│   ├── lstm_predictions.png
│   ├── seq2seq_cnn_predictions.png
│   ├── model_comparison.png
│   └── results_summary.json
│
└── 
```

## 🚀 Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd wikipedia-traffic-forecasting
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

1. Go to [Kaggle Competition Page](https://www.kaggle.com/c/web-traffic-time-series-forecasting/data)
2. Download `train_2.csv`
3. Place it in `data/` directory

```bash
mkdir data
# Move downloaded train_2.csv to data/ directory
```

## 🎮 Running the Code

### Quick Start - Run Full Experiment

```bash
python main_train_eval.py
```

This will:
1. Load and preprocess data
2. Train all three models
3. Generate predictions
4. Compare results and save visualizations

### Run Individual Models

**KNN Baseline:**
```python
from data_processor import WikiTrafficDataProcessor
from knn_model import KNNTimeSeriesForecaster

# Load and preprocess
processor = WikiTrafficDataProcessor('data/train_2.csv')
data = processor.load_data()
train_data, test_data, _ = processor.preprocess_for_lstm()

# Train KNN
knn = KNNTimeSeriesForecaster(k=10, distance_metric='canberra')
knn.train(train_data)
predictions = knn.predict(train_data, forecast_horizon=64)
```

**LSTM Model:**
```python
from lstm_model import LSTMTimeSeriesForecaster

# Initialize LSTM
lstm = LSTMTimeSeriesForecaster(
    n_layers=10,
    hidden_size=80,
    sequence_length=36,
    forecast_horizon=64
)

# Train
lstm.train(train_data, val_data=val_data, epochs=50)

# Predict
predictions = lstm.predict(test_data)
```

**Seq2Seq CNN:**
```python
from seq2seq_cnn import Seq2SeqCNNForecaster

# Initialize
cnn = Seq2SeqCNNForecaster(
    n_encoder_layers=8,
    input_length=128,
    output_length=64
)

# Train
cnn.train(train_data, val_data=val_data, epochs=50)

# Predict
predictions = cnn.predict(test_data)
```

## 📈 Methodology

### 1. Data Preprocessing

**For LSTM**:
- Handle missing values (NaN → 0)
- Apply log transformation: `log(x + 1)`
- Normalize to zero mean and unit variance
- Extract features: day of week, autocorrelations, traffic median

**For Seq2Seq CNN**:
- Log transformation
- Zero-mean normalization
- Create conditional features: project, access type, agent, null indicator, log mean

### 2. Model Architectures

#### KNN (Baseline)
- Window size: 64 days
- k = 10 nearest neighbors
- Distance metric: Canberra (best for positive, wide-variance data)
- Weighted average using inverse distance

#### LSTM
- 10-layer deep LSTM
- Hidden size: 80 units
- Sequence length: 36 steps (backpropagation length)
- Dropout: 0.4
- Learning rate: 1.0 → 0.2 (decay)
- Batch size: 1024

#### Seq2Seq CNN
- 8 encoder layers + 8 decoder layers
- Dilated causal convolutions (dilations: 1, 2, 4, 8, 16, 32, 64, 128)
- Receptive field: 128 steps
- Gated activation units
- Residual and skip connections
- Batch size: 128
- Learning rate: 0.001

### 3. Evaluation Metric

**SMAPE (Symmetric Mean Absolute Percentage Error)**:

```
SMAPE = (1/n) × Σ |F_t - A_t| / ((|F_t| + |A_t|) / 2) × 100
```

Where:
- F_t = Forecast value at time t
- A_t = Actual value at time t

**Benefits**:
- Robust to outliers
- Unified scale across different time series
- Symmetric (penalizes over/under-predictions equally)

## 📊 Results

### Model Performance Comparison

| Model | SMAPE | Training Time |
|-------|-------|---------------|
| KNN | 165.51 | ~1 hour |
| LSTM | 143.12 | ~24 hours |
| **Seq2Seq CNN** | **42.37** | ~12 hours |

**Key Findings**:
- Seq2Seq CNN significantly outperforms other approaches
- CNN architecture captures temporal dependencies effectively
- Causal convolutions enable parallel training (faster than LSTM)
- All models struggle with outlier spikes (external events)
- Missing values introduce prediction uncertainty

## 🔍 Implementation Details

### KNN
- Uses sliding window to create training samples
- Scikit-learn's KNeighborsRegressor with custom distance metrics
- Recursive forecasting for multi-step prediction
- Memory-efficient batch processing

### LSTM
- TensorFlow/Keras implementation
- Multiple LSTM layers with dropout regularization
- Exponential learning rate decay
- Early stopping and learning rate reduction callbacks
- Predicts entire forecast horizon in one pass

### Seq2Seq CNN
- Custom TensorFlow/Keras layers:
  - CausalConv1D (ensures causality)
  - GatedActivation (tanh-sigmoid gating)
  - ResidualBlock (with skip connections)
- Encoder-decoder architecture
- Parallelizable training (vs sequential RNN)

##  Key Concepts Explained

### 1. Causal Convolutions
- Ensures prediction at time t only uses data up to time t
- Prevents "information leakage" from future
- Implemented via left-padding

### 2. Dilated Convolutions
- Exponentially increases receptive field
- Layer i has dilation 2^i
- Captures long-range dependencies efficiently

### 3. Gated Activation
- `z = tanh(W_f * x) ⊙ σ(W_g * x)`
- Controls information flow
- Similar to LSTM gates but in CNN context

### 4. Residual Connections
- Helps gradient flow in deep networks
- Enables training of 8+ layer networks
- `output = input + F(input)`

## ⚙️ Hyperparameter Tuning

### Validation Strategy
- Last 64 days of training data used for validation
- Forward out-of-sample testing (not cross-validation)
- Early stopping based on validation SMAPE

### Key Hyperparameters

**KNN**:
- k ∈ {5, 10, 15, 20} → k=10 best
- Distance: Canberra best for this dataset

**LSTM**:
- Sequence length: 30-36 steps optimal
- State size: 45-50 optimal
- Dropout: 0.4 optimal

**Seq2Seq CNN**:
- Fixed output length: 64
- 8 layers empirically best
- 32 neurons in dense layers

##  Learning Outcomes

### Mathematical Concepts
- Time series analysis and forecasting
- Symmetric error metrics (SMAPE)
- Autocorrelation for feature extraction
- Log transformation for variance stabilization
- Fourier analysis for periodicity detection

### Machine Learning Concepts
- K-Nearest Neighbors for regression
- Recurrent Neural Networks (LSTM)
- Convolutional Neural Networks for sequences
- Encoder-decoder architectures
- Attention mechanisms and skip connections
- Regularization (dropout)
- Learning rate scheduling

### Deep Learning Implementation
- TensorFlow/Keras model building
- Custom layer implementation
- Training loop and callbacks
- Hyperparameter optimization
- Model evaluation and comparison

## 🚧 Challenges and Limitations

### Challenges Addressed
1. **Missing Values**: Treated as NaN, encoded as feature
2. **Scale Variance**: Log transformation + normalization
3. **Outlier Spikes**: SMAPE metric more robust than MSE
4. **Long-term Dependencies**: LSTM cells, dilated convolutions
5. **Computational Cost**: Batch processing, GPU acceleration

### Limitations
1. **Outlier Prediction**: Models cannot predict spikes from external events
2. **Missing Data**: Zero vs actual zero ambiguity
3. **Computational Resources**: LSTM training takes ~24 hours
4. **Seasonality**: Not all series have clear patterns
5. **External Features**: No news/events data incorporated

##  Future Work

1. **Model Improvements**:
   - Implement Seq2Seq LSTM with attention
   - Add DenseNet-style skip connections
   - Explore GRU-based variants
   - Ensemble methods combining all models

2. **Feature Engineering**:
   - Incorporate external data (news events, holidays)
   - Weather data for location-based pages
   - Cross-page correlation features
   - Text features from page titles

3. **Advanced Techniques**:
   - Transformer-based models
   - Prophet for seasonality decomposition
   - Neural basis expansion analysis (N-BEATS)
   - Transfer learning across similar pages

4. **Optimization**:
   - Bayesian hyperparameter optimization
   - Mixed precision training
   - Model quantization for deployment
   - Real-time inference optimization

##  References

1. [Kaggle Competition Dataset](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
2. Cho et al. (2014) - "Learning Phrase Representations using RNN Encoder-Decoder"
3. Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
4. Vaswani et al. (2017) - "Attention Is All You Need"
5. Van den Oord et al. (2016) - "WaveNet: A Generative Model for Raw Audio"
6. Gehring et al. (2017) - "Convolutional Sequence to Sequence Learning"
7. Karim et al. (2019) - "LSTM Fully Convolutional Networks for Time Series Classification"

##  Contributing

For course project purposes, this repository is maintained by the project team. For suggestions:

1. Open an issue describing the problem/enhancement
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

##  License

This project is for educational purposes as part of UE23CS352A coursework.

##  Authors

**Course**: UE23CS352A - Machine Learning  
**Institution**: PES University 
**Semester**: 5




## 🎯 Quick Command Reference

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full experiment
python main_train_eval.py

# Run with custom epochs
python -c "from main_train_eval import ExperimentRunner; \
           runner = ExperimentRunner('data/train_1.csv'); \
           runner.run_full_experiment(train_lstm_epochs=50, train_cnn_epochs=50)"


