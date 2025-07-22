import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from config import *
import os
from train import train_mavae

class MAVAEVisualizer:
    """Visualization tools for MA-VAE model"""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
        """
        Plot training history metrics
        
        Args:
            history: Training history dictionary
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(131)
        plt.plot(history['loss'], label='Training')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot reconstruction loss
        plt.subplot(132)
        plt.plot(history['log_probs_loss'], label='Training')
        if 'val_log_probs_loss' in history:
            plt.plot(history['val_log_probs_loss'], label='Validation')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot KL loss
        plt.subplot(133)
        plt.plot(history['kl_loss'], label='Training')
        if 'val_kl_loss' in history:
            plt.plot(history['val_kl_loss'], label='Validation')
        plt.title('KL Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/training_history.png")
        plt.show()
    
    @staticmethod
    def plot_reconstruction(model: tf.keras.Model,
                          original: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None):
        """
        Plot original vs reconstructed sequences
        
        Args:
            model: Trained MA-VAE model
            original: Original sequence data
            feature_names: Optional list of feature names
            save_path: Optional path to save the plot
        """
        # Get reconstruction
        reconstruction = model.predict(original[np.newaxis])[0]  # Use mean
        
        n_features = min(6, original.shape[-1])  # Plot up to 6 features
        plt.figure(figsize=(15, 2*n_features))
        
        for i in range(n_features):
            plt.subplot(n_features, 1, i+1)
            plt.plot(original[:, i], label='Original', alpha=0.7)
            plt.plot(reconstruction[:, i], label='Reconstructed', alpha=0.7)
            if feature_names and i < len(feature_names):
                plt.title(f'Feature: {feature_names[i]}')
            else:
                plt.title(f'Feature {i+1}')
            plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/reconstruction.png")
        plt.show()
    
    @staticmethod
    def plot_latent_space(model: tf.keras.Model,
                         data: np.ndarray,
                         labels: Optional[np.ndarray] = None,
                         method: str = 'tsne',
                         save_path: Optional[str] = None):
        """
        Visualize latent space using dimensionality reduction
        
        Args:
            model: Trained MA-VAE model
            data: Input data
            labels: Optional labels for coloring (0 for normal, 1 for anomaly)
            method: Dimensionality reduction method ('tsne' or 'pca')
            save_path: Optional path to save the plot
        """
        # Get latent representations
        z_mean = model.predict(data)[3]  # Get z_mean from model output
        
        # Reshape to 2D if needed
        if len(z_mean.shape) > 2:
            z_mean = z_mean.reshape(z_mean.shape[0], -1)
        
        # Reduce dimensionality
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=RANDOM_SEED)
        else:
            reducer = PCA(n_components=2, random_state=RANDOM_SEED)
        
        z_2d = reducer.fit_transform(z_mean)
        
        # Plot
        plt.figure(figsize=(10, 8))
        if labels is not None:
            scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='viridis')
            plt.colorbar(scatter, label='Label')
        else:
            plt.scatter(z_2d[:, 0], z_2d[:, 1])
        
        plt.title(f'Latent Space Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if save_path:
            plt.savefig(f"{save_path}/latent_space_{method}.png")
        plt.show()
    
    @staticmethod
    def plot_attention_weights(model: tf.keras.Model,
                             sequence: np.ndarray,
                             save_path: Optional[str] = None):
        """
        Visualize attention weights for a sequence
        
        Args:
            model: Trained MA-VAE model
            sequence: Input sequence
            save_path: Optional path to save the plot
        """
        # Get attention weights
        attention = model.predict(sequence[np.newaxis])[6]  # Get attention matrix
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention[0], cmap='viridis')
        plt.title('Attention Weights')
        plt.xlabel('Time Step (Key)')
        plt.ylabel('Time Step (Query)')
        
        if save_path:
            plt.savefig(f"{save_path}/attention_weights.png")
        plt.show()
    
    @staticmethod
    def plot_anomaly_scores(normal_scores: np.ndarray,
                           abnormal_scores: np.ndarray,
                           threshold: float,
                           save_path: Optional[str] = None):
        """
        Plot distribution of anomaly scores
        
        Args:
            normal_scores: Anomaly scores for normal sequences
            abnormal_scores: Anomaly scores for abnormal sequences
            threshold: Anomaly threshold
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot distributions
        plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal', density=True)
        plt.hist(abnormal_scores, bins=30, alpha=0.5, label='Abnormal', density=True)
        
        # Plot threshold
        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
        
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
        
        if save_path:
            plt.savefig(f"{save_path}/anomaly_scores.png")
        plt.show()

# Example usage
if __name__ == "__main__":
    visualizer = MAVAEVisualizer()
    
    # Check if model and history exist, if not, train the model first
    if not os.path.exists('output/model') or not os.path.exists('output/history.npy'):
        print("Model or history not found. Training model first...")
        model, history = train_mavae(
            data_path="data.mat",
            save_path="output",
            n_features=REDUCED_FEATURES
        )
    else:
        # Load model and data
        model = tf.keras.models.load_model('output/model')
        history = np.load('output/history.npy', allow_pickle=True).item()
    
    # Plot training history
    visualizer.plot_training_history(history, save_path='output')
    
    # Check if example sequence exists
    try:
        sequence = np.load('output/example_sequence.npy')
        visualizer.plot_reconstruction(model, sequence, save_path='output')
    except FileNotFoundError:
        print("Example sequence not found, skipping reconstruction plot")
    
    # Check if test data exists
    try:
        data = np.load('output/test_data.npy')
        labels = np.load('output/test_labels.npy')
        visualizer.plot_latent_space(model, data, labels, save_path='output')
    except FileNotFoundError:
        print("Test data not found, skipping latent space plot")
    
    # Check if sequence exists for attention weights
    try:
        sequence = np.load('output/example_sequence.npy')
        visualizer.plot_attention_weights(model, sequence, save_path='output')
    except FileNotFoundError:
        print("Example sequence not found, skipping attention weights plot")
    
    # Check if anomaly scores exist
    try:
        normal_scores = np.load('output/normal_scores.npy')
        abnormal_scores = np.load('output/abnormal_scores.npy')
        threshold = np.load('output/threshold.npy')
        visualizer.plot_anomaly_scores(normal_scores, abnormal_scores, threshold, save_path='output')
    except FileNotFoundError:
        print("Anomaly scores not found, skipping anomaly scores plot") 