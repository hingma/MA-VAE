import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import precision_score, recall_score, f1_score, auc
from typing import Tuple, Dict, Optional
from config import *
from data_process import WaferDataProcessor
from visualize import MAVAEVisualizer
import os

def evaluate_sequence(model: tf.keras.Model,
                     data: np.ndarray,
                     window_size: int = WINDOW_SIZE,
                     shift: int = 1) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Evaluate a sequence using the model
    
    Args:
        model: Trained MA-VAE model
        data: Input sequence
        window_size: Size of sliding windows
        shift: Window shift size
        
    Returns:
        Tuple of (anomaly_scores, reconstruction_params)
    """
    print(f"\nEvaluating sequence of shape {data.shape}...")
    
    # Create windows
    processor = WaferDataProcessor()
    windows, _ = processor.create_windows(
        data,
        window_size=window_size,
        window_shift=shift
    )
    
    print(f"Created {len(windows)} windows of size {window_size}")
    
    # Get model predictions
    reconstruction = model.predict(
        windows,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    
    # Extract parameters
    mean = reconstruction[0]
    log_var = reconstruction[1]
    var = tf.math.exp(log_var)
    
    # Reverse window the parameters
    def rev_window(windows: np.ndarray, shift: int) -> np.ndarray:
        """Reverse windowing using mean of overlapping regions"""
        n_windows = windows.shape[0]
        seq_len = (n_windows - 1) * shift + windows.shape[1]
        n_features = windows.shape[2]
        
        # Initialize output array with NaN
        data = np.zeros((n_windows, seq_len, n_features)) + np.nan
        
        # Place each window in the correct position
        for i in range(n_windows):
            data[i, i:i + windows.shape[1], :] = windows[i]
            
        # Take mean across windows (ignoring NaN)
        return np.nanmean(data, axis=0)
    
    # Reverse window parameters
    mean_seq = rev_window(mean, shift)
    var_seq = rev_window(var, shift)
    std_seq = np.sqrt(var_seq)
    
    # Calculate anomaly scores
    output_dist = tfp.distributions.MultivariateNormalDiag(
        loc=mean_seq,
        scale_diag=std_seq
    )
    log_probs = -output_dist.unnormalized_log_prob(data).numpy()
    
    return log_probs, {'mean': mean_seq, 'std': std_seq}

def evaluate_model(model: tf.keras.Model,
                  normal_data: np.ndarray,
                  abnormal_data: np.ndarray,
                  threshold_percentile: float = 95.0,
                  save_path: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate model performance
    
    Args:
        model: Trained MA-VAE model
        normal_data: Normal test sequences
        abnormal_data: Abnormal test sequences
        threshold_percentile: Percentile for anomaly threshold
        save_path: Optional path to save results and visualizations
        
    Returns:
        Dictionary of performance metrics
    """
    print("\n=== Starting Model Evaluation ===")
    print(f"Normal sequences: {len(normal_data)}")
    print(f"Abnormal sequences: {len(abnormal_data)}")
    
    # Evaluate normal sequences
    print("\nEvaluating normal sequences...")
    normal_scores = []
    normal_reconstructions = []
    for i, sequence in enumerate(normal_data):
        print(f"Processing normal sequence {i+1}/{len(normal_data)}")
        scores, recon = evaluate_sequence(model, sequence)
        normal_scores.append(np.max(scores))  # Use max score for sequence
        normal_reconstructions.append(recon['mean'])
    normal_scores = np.array(normal_scores)
    
    # Evaluate abnormal sequences
    print("\nEvaluating abnormal sequences...")
    abnormal_scores = []
    abnormal_reconstructions = []
    for i, sequence in enumerate(abnormal_data):
        print(f"Processing abnormal sequence {i+1}/{len(abnormal_data)}")
        scores, recon = evaluate_sequence(model, sequence)
        abnormal_scores.append(np.max(scores))
        abnormal_reconstructions.append(recon['mean'])
    abnormal_scores = np.array(abnormal_scores)
    
    # Set threshold
    threshold = np.percentile(normal_scores, threshold_percentile)
    print(f"\nAnomaly threshold: {threshold:.4f} (percentile: {threshold_percentile})")
    
    # Get predictions
    normal_preds = normal_scores >= threshold
    abnormal_preds = abnormal_scores >= threshold
    
    # Create true labels
    true_labels = np.concatenate([
        np.zeros_like(normal_scores),
        np.ones_like(abnormal_scores)
    ])
    
    # Create predicted labels
    pred_labels = np.concatenate([normal_preds, abnormal_preds])
    
    # Calculate metrics
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    # Calculate PR-AUC
    all_scores = np.concatenate([normal_scores, abnormal_scores])
    thresholds = np.percentile(all_scores, np.arange(0, 100.1, 0.1))
    
    precisions = []
    recalls = []
    for thresh in thresholds:
        preds = all_scores >= thresh
        precisions.append(precision_score(true_labels, preds))
        recalls.append(recall_score(true_labels, preds))
    
    pr_auc = auc(recalls, precisions)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Save results if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save scores and metrics
        np.save(f"{save_path}/normal_scores.npy", normal_scores)
        np.save(f"{save_path}/abnormal_scores.npy", abnormal_scores)
        np.save(f"{save_path}/threshold.npy", threshold)
        np.save(f"{save_path}/metrics.npy", {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'pr_auc': pr_auc
        })
        
        # Save example reconstructions
        np.save(f"{save_path}/example_normal.npy", normal_data[0])
        np.save(f"{save_path}/example_abnormal.npy", abnormal_data[0])
        
        # Create visualizations
        visualizer = MAVAEVisualizer()
        
        # Plot anomaly score distributions
        visualizer.plot_anomaly_scores(
            normal_scores,
            abnormal_scores,
            threshold,
            save_path=save_path
        )
        
        # Plot example reconstructions
        visualizer.plot_reconstruction(
            model,
            normal_data[0],
            save_path=save_path
        )
        
        # Plot latent space
        all_data = np.concatenate([normal_data, abnormal_data])
        labels = np.concatenate([
            np.zeros(len(normal_data)),
            np.ones(len(abnormal_data))
        ])
        visualizer.plot_latent_space(
            model,
            all_data,
            labels,
            save_path=save_path
        )
        
        # Plot attention weights
        visualizer.plot_attention_weights(
            model,
            normal_data[0],
            save_path=save_path
        )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pr_auc': pr_auc,
        'threshold': threshold
    }

if __name__ == "__main__":
    print("\nLoading model and data...")
    model = tf.keras.models.load_model('output/model')
    
    # Load test data
    processor = WaferDataProcessor()
    _, _, abnormal_data = processor.process_data("data.mat")
    
    # Split normal data into train/test
    normal_data = np.load("output/normal.npy")
    test_size = int(0.2 * len(normal_data))
    normal_test = normal_data[-test_size:]
    
    # Evaluate
    metrics = evaluate_model(
        model,
        normal_test,
        abnormal_data,
        save_path="output/evaluation"
    ) 