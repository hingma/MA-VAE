import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import scipy.io
from typing import Tuple, List, Optional
from config import *

class WaferDataProcessor:
    def __init__(self, n_features: int = N_FEATURES):
        """
        Initialize the data processor
        
        Args:
            n_features: Number of features to use (for dimension reduction)
        """
        self.n_features = n_features
        self.scaler = StandardScaler()
        
    def load_mat_data(self, mat_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load and reshape data from .mat file
        
        Args:
            mat_path: Path to the .mat file
            
        Returns:
            Tuple of (normal_samples, abnormal_samples) as lists of numpy arrays
        """
        mat_data = scipy.io.loadmat(mat_path)
        raw_data = list(mat_data.values())[3]  # Skip built-in keys
        n_samples = raw_data.shape[0]
        
        # Reshape samples
        samples = []
        for i in range(n_samples):
            # Each sample is already a 2D array with shape (time_steps, features)
            sample = raw_data[i, 0]  # Get the actual array data
            if sample.shape[1] != self.n_features:
                # If needed, select first n_features columns
                sample = sample[:, :self.n_features]
            samples.append(sample)
        
        # Split into normal and abnormal
        normal_samples = samples[:46]  # First 46 are normal
        abnormal_samples = samples[46:]  # Last 6 are abnormal
        
        return normal_samples, abnormal_samples
    
    def standardize_and_pad(self, 
                           normal_samples: List[np.ndarray], 
                           abnormal_samples: Optional[List[np.ndarray]] = None
                           ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Standardize and zero-pad samples to same length
        
        Args:
            normal_samples: List of normal samples for fitting scaler
            abnormal_samples: Optional list of abnormal samples to transform
            
        Returns:
            Tuple of processed (normal_samples, abnormal_samples) as numpy arrays
        """
        # Convert to float32 and concatenate all normal samples for fitting scaler
        normal_data_concat = np.vstack([sample.astype(np.float32) for sample in normal_samples])
        self.scaler.fit(normal_data_concat)
        
        def process_sample(sample: np.ndarray) -> np.ndarray:
            # Convert to float32 and standardize
            sample = sample.astype(np.float32)
            sample_scaled = self.scaler.transform(sample)
            # Zero pad to max length
            current_len = sample_scaled.shape[0]
            if current_len < SEQUENCE_LENGTH:
                padding = np.zeros((SEQUENCE_LENGTH - current_len, sample_scaled.shape[1]), dtype=np.float32)
                sample_padded = np.vstack([sample_scaled, padding])
            else:
                sample_padded = sample_scaled[:SEQUENCE_LENGTH]
            return sample_padded
        
        # Process normal samples
        normal_processed = np.array([process_sample(s) for s in normal_samples])
        
        # Process abnormal samples if provided
        abnormal_processed = None
        if abnormal_samples is not None:
            abnormal_processed = np.array([process_sample(s) for s in abnormal_samples])
            
        return normal_processed, abnormal_processed
    
    def create_windows(self, 
                      data: np.ndarray, 
                      window_size: int = WINDOW_SIZE, 
                      window_shift: int = WINDOW_SHIFT
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from data
        
        Args:
            data: Input data to window
            window_size: Size of each window
            window_shift: Shift between windows
            
        Returns:
            Tuple of (windows, labels) where labels are same as windows for reconstruction
        """
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
        """
        Prepare TensorFlow datasets for training
        
        Args:
            normal_samples: Processed normal samples
            val_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Create windows
        windows, labels = self.create_windows(normal_samples)
        
        # Convert to float32 tensors
        windows = tf.cast(windows, tf.float32)
        labels = tf.cast(labels, tf.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((windows, labels))
        
        # Split into train and validation
        val_size = int(val_split * len(windows))
        val_data = dataset.take(val_size)
        train_data = dataset.skip(val_size)
        
        # Batch and shuffle
        train_data = train_data.shuffle(len(windows)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_data = val_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        return train_data, val_data
    
    def process_data(self, 
                    mat_path: str,
                    save_path: Optional[str] = None
                    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray]:
        """
        Complete data processing pipeline
        
        Args:
            mat_path: Path to .mat file
            save_path: Optional path to save processed datasets
            
        Returns:
            Tuple of (train_dataset, val_dataset, abnormal_processed)
        """
        # Load data
        normal_samples, abnormal_samples = self.load_mat_data(mat_path)
        
        # Standardize and pad
        normal_processed, abnormal_processed = self.standardize_and_pad(normal_samples, abnormal_samples)
        
        # Create TF datasets
        train_data, val_data = self.prepare_tf_datasets(normal_processed)
        
        # Save if path provided
        if save_path:
            train_data.save(f"{save_path}/train")
            val_data.save(f"{save_path}/val")
            np.save(f"{save_path}/abnormal.npy", abnormal_processed)
            np.save(f"{save_path}/scaler.npy", 
                   {'mean': self.scaler.mean_, 'scale': self.scaler.scale_})
        
        return train_data, val_data, abnormal_processed
