"""
Sequence-to-Sequence with Causal Dilated CNN
Implements WaveNet-style architecture for time series forecasting
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

class CausalConv1D(layers.Layer):
    """Causal 1D Convolution - ensures no future information leakage"""
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = (kernel_size - 1) * dilation_rate
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='valid',
            dilation_rate=dilation_rate
        )
    
    def call(self, inputs):
        padded = tf.pad(inputs, [[0, 0], [self.padding, 0], [0, 0]])
        output = self.conv(padded)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })
        return config

class GatedActivation(layers.Layer):
    """Gated Activation: z = tanh(W_f * x) ⊙ σ(W_g * x)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        filter_half = inputs[:, :, :inputs.shape[2]//2]
        gate_half = inputs[:, :, inputs.shape[2]//2:]
        tanh_out = tf.nn.tanh(filter_half)
        sigmoid_out = tf.nn.sigmoid(gate_half)
        return tanh_out * sigmoid_out

class ResidualBlock(layers.Layer):
    """Residual block with dilated causal convolution - FIXED VERSION"""
    def __init__(self, filters, kernel_size, dilation_rate, 
                 residual_channels=32, skip_channels=32, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        
        # Fixed layers - created once in __init__
        self.causal_conv = CausalConv1D(
            filters=filters * 2,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate
        )
        self.gated_activation = GatedActivation()
        self.residual_dense = layers.Conv1D(residual_channels, 1)
        self.skip_dense = layers.Conv1D(skip_channels, 1)
        # ✅ Pre-create projection layer to match input channels
        self.input_projection = layers.Conv1D(residual_channels, 1)

    def call(self, inputs):
        x = self.causal_conv(inputs)
        x = self.gated_activation(x)
        residual = self.residual_dense(x)
        
        # ✅ Always project inputs to match residual_channels
        # No dynamic layer creation inside call()
        inputs_matched = self.input_projection(inputs)
        
        # Ensure same sequence length (handle edge cases)
        min_len = tf.minimum(tf.shape(inputs_matched)[1], tf.shape(residual)[1])
        inputs_matched = inputs_matched[:, :min_len, :]
        residual = residual[:, :min_len, :]
        
        residual_out = inputs_matched + residual
        skip = self.skip_dense(x)
        return residual_out, skip

class Seq2SeqCNN(keras.Model):
    """Sequence-to-Sequence model with Causal Dilated CNN"""
    def __init__(self, n_encoder_layers=8, n_decoder_layers=8, filters=32,
                 kernel_size=2, input_length=128, output_length=64,
                 conditional_features_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_length = input_length
        self.output_length = output_length
        self.conditional_features_dim = conditional_features_dim
        
        # ✅ Pass channel dimensions to ResidualBlocks
        self.encoder_blocks = []
        for i in range(n_encoder_layers):
            dilation = 2 ** i
            self.encoder_blocks.append(
                ResidualBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    residual_channels=filters,  # ✅ Match filters
                    skip_channels=filters,      # ✅ Match filters
                    name=f'encoder_block_{i+1}'
                )
            )
        
        self.decoder_blocks = []
        for i in range(n_decoder_layers):
            dilation = 2 ** i
            self.decoder_blocks.append(
                ResidualBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    residual_channels=filters,  # ✅ Match filters
                    skip_channels=filters,      # ✅ Match filters
                    name=f'decoder_block_{i+1}'
                )
            )
        
        self.output_projection = layers.Conv1D(1, 1, name='output_projection')

    def call(self, inputs, training=False):
        if isinstance(inputs, list):
            encoder_input = inputs[0]
            conditional_features = inputs[1] if len(inputs) > 1 else None
        else:
            encoder_input = inputs
            conditional_features = None
        
        skip_connections = []
        x = encoder_input
        for block in self.encoder_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        encoder_output = tf.add_n(skip_connections)
        batch_size = tf.shape(encoder_input)[0]
        decoder_input = tf.zeros((batch_size, self.output_length, self.filters))
        encoder_broadcast = tf.tile(
            tf.expand_dims(tf.reduce_mean(encoder_output, axis=1), axis=1),
            [1, self.output_length, 1]
        )
        decoder_input = decoder_input + encoder_broadcast
        
        skip_connections = []
        x = decoder_input
        for block in self.decoder_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        decoder_output = tf.add_n(skip_connections)
        output = self.output_projection(decoder_output)
        output = tf.squeeze(output, axis=-1)
        return output

class Seq2SeqCNNForecaster:
    """Wrapper for Seq2Seq CNN forecasting"""
    def __init__(self, n_encoder_layers=8, n_decoder_layers=8, filters=32,
                 input_length=128, output_length=64, batch_size=128,
                 learning_rate=0.001, early_stop_patience=10):
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.filters = filters
        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_patience = early_stop_patience
        self.model = None
        self.history = None

    def build_model(self, conditional_features_dim=0):
        """Build and compile model"""
        print("\nBuilding Seq2Seq CNN model:")
        print(f"  Encoder layers: {self.n_encoder_layers}")
        print(f"  Decoder layers: {self.n_decoder_layers}")
        print(f"  Filters: {self.filters}")
        print(f"  Input length: {self.input_length}")
        print(f"  Output length: {self.output_length}")
        
        self.model = Seq2SeqCNN(
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            filters=self.filters,
            input_length=self.input_length,
            output_length=self.output_length,
            conditional_features_dim=conditional_features_dim
        )
        
        # Build model with dummy input
        dummy_input = tf.random.normal((1, self.input_length, 1))
        _ = self.model(dummy_input)
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        print("\nModel built successfully!")
        self.model.summary()
        return self.model

    def create_sequences(self, data):
        """Create input-output sequences"""
        n_series, n_timesteps = data.shape
        X, y = [], []
        for i in range(n_series):
            series = data[i]
            for t in range(self.input_length, n_timesteps - self.output_length + 1):
                X.append(series[t-self.input_length:t])
                y.append(series[t:t+self.output_length])
        X = np.array(X).reshape(-1, self.input_length, 1)
        y = np.array(y)
        return X, y

    def train(self, train_data, val_data=None, epochs=50, verbose=1):
        """Train model"""
        print("\n" + "="*60)
        print("Training Seq2Seq CNN Model")
        print("="*60)
        print("Creating training sequences...")
        X_train, y_train = self.create_sequences(train_data)
        print(f"Training samples: {len(X_train)}")
        
        if self.model is None:
            self.build_model()
        
        validation_data = None
        if val_data is not None:
            print("Creating validation sequences...")
            X_val, y_val = self.create_sequences(val_data)
            validation_data = (X_val, y_val)
            print(f"Validation samples: {len(X_val)}")
        
        callback_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_data is not None else 'loss',
                patience=self.early_stop_patience,
                restore_best_weights=True
            )
        ]
        
        print(f"\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=self.batch_size,
            epochs=epochs,
            callbacks=callback_list,
            verbose=verbose
        )
        print("\nTraining completed!")
        return self.history

    def predict(self, test_data):
        """Generate predictions"""
        print("\nGenerating predictions...")
        n_series = test_data.shape[0]
        predictions = []
        for i in range(n_series):
            input_seq = test_data[i, -self.input_length:]
            input_seq = input_seq.reshape(1, self.input_length, 1)
            pred = self.model.predict(input_seq, verbose=0)
            predictions.append(pred[0])
        return np.array(predictions)

    def evaluate_smape(self, predictions, actual):
        """Calculate SMAPE"""
        numerator = np.abs(predictions - actual)
        denominator = (np.abs(predictions) + np.abs(actual)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        smape = np.mean(numerator / denominator) * 100
        return smape

    def visualize_predictions(self, train_data, test_data, predictions,
                            num_samples=5, save_path='seq2seq_cnn_predictions.png'):
        """Visualize predictions"""
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        for i in range(num_samples):
            idx = np.random.randint(0, len(predictions))
            train_series = train_data[idx, -100:]
            train_x = np.arange(len(train_series))
            axes[i].plot(train_x, train_series, 'b-', label='Training', alpha=0.6)
            test_series = test_data[idx, :len(predictions[idx])]
            test_x = np.arange(len(train_series), len(train_series) + len(test_series))
            axes[i].plot(test_x, test_series, 'g-', label='Actual', linewidth=2)
            pred_series = predictions[idx]
            axes[i].plot(test_x, pred_series, 'r--', label='Seq2Seq CNN', linewidth=2)
            axes[i].set_title(f'Series {idx} - Seq2Seq CNN Forecast')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Normalized Traffic')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.show()