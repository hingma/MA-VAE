import tensorflow as tf
from typing import Optional, Tuple
from config import *
from data_manager import DataManager
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from attention import MA
from ma_vae import MA_VAE
import numpy as np
import time
import os

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Callback for detailed training progress reporting"""
    def __init__(self):
        super(TrainingProgressCallback, self).__init__()
        self.epoch_start_time = None
        self.training_start_time = None
    
    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        print("\n=== Starting MA-VAE Training ===")
        print(f"Model Configuration:")
        print(f"- Input Features: {REDUCED_FEATURES}")
        print(f"- Sequence Length: {SEQUENCE_LENGTH}")
        print(f"- Window Size: {WINDOW_SIZE}")
        print(f"- Latent Dimension: {LATENT_DIM}")
        print(f"- Batch Size: {BATCH_SIZE}")
        print("==============================\n")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.training_start_time
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"- Time: {epoch_time:.2f}s (Total: {total_time:.2f}s)")
        print(f"- Loss: {logs['loss']:.4f}")
        print(f"- Reconstruction Loss: {logs['log_probs_loss']:.4f}")
        print(f"- KL Loss: {logs['kl_loss']:.4f}")
        if 'val_loss' in logs:
            print(f"- Validation Loss: {logs['val_loss']:.4f}")
            print(f"- Val Reconstruction Loss: {logs['val_log_probs_loss']:.4f}")
            print(f"- Val KL Loss: {logs['val_kl_loss']:.4f}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.training_start_time
        print("\n=== Training Completed ===")
        print(f"Total training time: {total_time:.2f}s")
        print("=========================\n")

class VAE_kl_annealing(tf.keras.callbacks.Callback):
    """KL annealing callback"""
    def __init__(self,
                 annealing_epochs: int = ANNEALING_EPOCHS,
                 type: str = "cyclical",
                 grace_period: int = GRACE_PERIOD,
                 start: float = INITIAL_BETA,
                 end: float = 1e-2,
                 lower_initial_betas: bool = False):
        super(VAE_kl_annealing, self).__init__()
        self.annealing_epochs = annealing_epochs
        self.type = type
        self.grace_period = grace_period
        self.grace_period_idx = max(0, grace_period - 1)
        self.start = start
        self.end = end
        
        if type in ["cyclical", "monotonic"]:
            self.beta_values = np.linspace(start, end, annealing_epochs)
            if lower_initial_betas:
                self.beta_values[:annealing_epochs // 2] /= 2

    def on_epoch_begin(self, epoch, logs=None):
        shifted_epochs = tf.math.maximum(0.0, epoch - self.grace_period_idx)
        
        if epoch < self.grace_period_idx or self.type == "normal":
            step_size = self.start / self.grace_period
            new_value = step_size * (epoch % self.grace_period)
        elif self.type == "monotonic":
            new_value = self.beta_values[min(epoch, self.annealing_epochs - 1)]
        elif self.type == "cyclical":
            new_value = self.beta_values[int(shifted_epochs % self.annealing_epochs)]
            
        self.model.beta.assign(new_value)
        print(f"Beta value: {self.model.beta.numpy():.10f}, "
              f"cycle epoch {int(shifted_epochs % self.annealing_epochs)}")

def train_mavae(data_path: str,
                output_dir: str = "output",
                force_reprocess: bool = False) -> Tuple[MA_VAE, dict]:
    """
    Train the MA-VAE model
    
    Args:
        data_path: Path to .mat data file
        output_dir: Directory to save outputs
        force_reprocess: Whether to force data reprocessing
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    print("\nInitializing MA-VAE training...")
    
    # Initialize data manager
    data_manager = DataManager(output_dir)
    
    # Load or process data
    try:
        if force_reprocess:
            raise FileNotFoundError("Forcing data reprocessing")
        print("Loading processed data...")
        train_data, val_data, normal_data, abnormal_data = data_manager.load_processed_data()
    except FileNotFoundError:
        print(f"Processing data from: {data_path}")
        train_data, val_data, normal_data, abnormal_data = data_manager.process_data(data_path)
    
    print("\nData Ready:")
    print(f"- Training data shape: {train_data.element_spec[0].shape}")
    print(f"- Validation data shape: {val_data.element_spec[0].shape}")
    print(f"- Normal windows shape: {normal_data.shape}")
    print(f"- Abnormal data shape: {abnormal_data.shape}")
    
    print("\nBuilding model components...")
    # Create model components
    encoder = VAE_Encoder(
        seq_len=WINDOW_SIZE,
        latent_dim=LATENT_DIM,
        features=N_FEATURES
    )
    
    decoder = VAE_Decoder(
        seq_len=WINDOW_SIZE,
        latent_dim=LATENT_DIM,
        features=N_FEATURES
    )
    
    ma = MA(
        seq_len=WINDOW_SIZE,
        latent_dim=LATENT_DIM,
        features=N_FEATURES
    )
    
    # Create and compile model
    model = MA_VAE(encoder, decoder, ma, beta=INITIAL_BETA)
    
    # Build model with sample input
    sample_shape = (1, WINDOW_SIZE, N_FEATURES)
    sample_input = tf.zeros(sample_shape)
    _ = model(sample_input)
    
    optimizer = tf.keras.optimizers.legacy.Adam(amsgrad=True)
    model.compile(optimizer=optimizer)
    
    print("\nSetting up training...")
    # Setup callbacks
    callbacks = [
        TrainingProgressCallback(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_log_probs_loss',
            mode='min',
            verbose=1,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        VAE_kl_annealing(
            annealing_epochs=ANNEALING_EPOCHS,
            type="cyclical",
            grace_period=GRACE_PERIOD,
            start=INITIAL_BETA,
            end=1e-2
        )
    ]
    
    # Train model
    history = model.fit(
        train_data,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        validation_data=val_data,
        verbose=0  # Turn off default progress bar
    )
    
    # Save model
    model_path = os.path.join(output_dir, "model")
    print(f"\nSaving model to: {model_path}")
    model.save(model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, "history.npy")
    print(f"Saving training history to: {history_path}")
    np.save(history_path, history.history)
    
    return model, history.history

if __name__ == "__main__":
    # Train model
    model, history = train_mavae(
        data_path="data.mat",
        output_dir="output",
        force_reprocess=True # Set to True to force data reprocessing
    )