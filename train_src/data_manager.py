import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import scipy.io
from typing import Tuple, List, Optional, Dict
from config import *
import os

class DataManager:
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the data manager
        
        Args:
            output_dir: Directory to save/load processed data
        """
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        os.makedirs(output_dir, exist_ok=True)
        
    def load_mat_data(self, mat_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load raw data from .mat file"""
        mat_data = scipy.io.loadmat(mat_path)
        raw_data = list(mat_data.values())[3]  # Skip built-in keys
        
        # Reshape samples
        samples = []
        for i in range(raw_data.shape[0]):
            sample = raw_data[i, 0]  # Get the actual array data
            if sample.shape[1] != N_FEATURES:
                sample = sample[:, :N_FEATURES]
            samples.append(sample)
        
        # Split into normal and abnormal
        abnormal_indices = [1, 19, 26, 33, 41, 47]
        normal_samples = [s for i, s in enumerate(samples) if i not in abnormal_indices]
        abnormal_samples = [samples[i] for i in abnormal_indices]
        
        return normal_samples, abnormal_samples
    
    def standardize_and_pad(self, 
                           normal_samples: List[np.ndarray], 
                           abnormal_samples: Optional[List[np.ndarray]] = None,
                           fit_scaler: bool = True
                           ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Standardize and zero-pad samples"""
        if fit_scaler:
            # Fit scaler on normal data only
            normal_data_concat = np.vstack([s.astype(np.float32) for s in normal_samples])
            self.scaler.fit(normal_data_concat)
            self.save_scaler()
        
        def process_sample(sample: np.ndarray) -> np.ndarray:
            sample = sample.astype(np.float32)
            sample_scaled = self.scaler.transform(sample)
            current_len = sample_scaled.shape[0]
            if current_len < SEQUENCE_LENGTH:
                padding = np.zeros((SEQUENCE_LENGTH - current_len, sample_scaled.shape[1]))
                sample_padded = np.vstack([sample_scaled, padding])
            else:
                sample_padded = sample_scaled[:SEQUENCE_LENGTH]
            return sample_padded
        
        normal_processed = np.array([process_sample(s) for s in normal_samples])
        abnormal_processed = None
        if abnormal_samples is not None:
            abnormal_processed = np.array([process_sample(s) for s in abnormal_samples])
            
        return normal_processed, abnormal_processed
    
    def create_windows(self, 
                      data: np.ndarray, 
                      window_size: int = WINDOW_SIZE, 
                      window_shift: int = WINDOW_SHIFT
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows from data"""
        windows = []
        labels = []
        for sample in data:
            for i in range(0, len(sample) - window_size + 1, window_shift):
                window = sample[i:i + window_size]
                windows.append(window)
                labels.append(window)
        return np.array(windows), np.array(labels)
    
    def prepare_tf_datasets(self, 
                          normal_samples: np.ndarray,
                          val_split: float = 0.2
                          ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Prepare TensorFlow datasets for training"""
        windows, labels = self.create_windows(normal_samples)
        windows = tf.cast(windows, tf.float32)
        labels = tf.cast(labels, tf.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices((windows, labels))
        val_size = int(val_split * len(windows))
        
        val_data = dataset.take(val_size)
        train_data = dataset.skip(val_size)
        
        train_data = train_data.shuffle(len(windows)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_data = val_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        return train_data, val_data
    
    def save_scaler(self):
        """Save scaler parameters"""
        scaler_path = os.path.join(self.output_dir, "scaler.npy")
        np.save(scaler_path, {
            'mean': self.scaler.mean_,
            'scale': self.scaler.scale_
        })
    
    def load_scaler(self):
        """Load scaler parameters"""
        scaler_path = os.path.join(self.output_dir, "scaler.npy")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
        scaler_data = np.load(scaler_path, allow_pickle=True).item()
        self.scaler.mean_ = scaler_data['mean']
        self.scaler.scale_ = scaler_data['scale']
    
    def save_processed_data(self, 
                          train_data: tf.data.Dataset,
                          val_data: tf.data.Dataset,
                          normal_data: np.ndarray,
                          abnormal_data: np.ndarray):
        """Save all processed data"""
        # Save TF datasets
        train_data.save(os.path.join(self.output_dir, "train"))
        val_data.save(os.path.join(self.output_dir, "val"))
        
        # Save abnormal data
        np.save(os.path.join(self.output_dir, "abnormal.npy"), abnormal_data)
        
        # Save normal data
        np.save(os.path.join(self.output_dir, "normal.npy"), normal_data)
                    
        # Save scaler
        self.save_scaler()
    
    def load_processed_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
        """
        Load all processed data
        
        Returns:
            Tuple of (train_dataset, val_dataset, normal_data, abnormal_data)
        """
        train_data = tf.data.Dataset.load(os.path.join(self.output_dir, "train"))
        val_data = tf.data.Dataset.load(os.path.join(self.output_dir, "val"))
        normal_data = np.load(os.path.join(self.output_dir, "normal.npy"))
        abnormal_data = np.load(os.path.join(self.output_dir, "abnormal.npy"))
        self.load_scaler()
        return train_data, val_data, normal_data, abnormal_data
    
    def process_data(self, mat_path: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray]:
        """Complete data processing pipeline"""
        # Load and process data
        normal_samples, abnormal_samples = self.load_mat_data(mat_path)
        normal_processed, abnormal_processed = self.standardize_and_pad(normal_samples, abnormal_samples)
        # create windows
        train_data, val_data = self.prepare_tf_datasets(normal_processed)
        
        # Save processed data
        self.save_processed_data(train_data, val_data, normal_processed, abnormal_processed)
        
        return train_data, val_data, normal_processed, abnormal_processed

    def standardize_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Standardize a single sample using saved scaler
        
        Args:
            sample: 1D or 2D array of shape (n_features,) or (n_samples, n_features)
            
        Returns:
            Standardized sample with same shape as input
        """
        # Reshape 1D array to 2D if needed
        if len(sample.shape) == 1:
            sample_2d = sample.reshape(1, -1)
            return self.scaler.transform(sample_2d).reshape(-1)  # Transform and reshape back to 1D
        else:
            return self.scaler.transform(sample) 