import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional, Dict
from config import *
from data_manager import DataManager
from reportor import * 
import collections
import os
import json


CACHE_DIR = "output/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class MAVAEDetector:
    def __init__(self,
                 model_path: str,
                 data_manager: DataManager,
                 window_size: int = WINDOW_SIZE,
                 threshold_percentile: float = 95.0):
        """
        Initialize the MA-VAE detector
        
        Args:
            model_path: Path to the saved MA-VAE model
            data_manager: DataManager instance for data processing
            window_size: Size of sliding window for detection
            threshold_percentile: Percentile for anomaly threshold
        """
        print(f"\nLoading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        self.data_manager = data_manager
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.model_path = model_path
        
        # Initialize buffer for streaming data
        self.data_buffer = collections.deque(maxlen=window_size)
    
    def calculate_anomaly_score(self, window: np.ndarray) -> float:
        """
        Calculate anomaly score for a window of data
        
        Args:
            window: Window of standardized data
            
        Returns:
            Anomaly score for the window (normalized by window size and features)
        """
        try:
            # Ensure window has correct shape
            if len(window.shape) == 2:
                window_reshaped = window.reshape(1, self.window_size, -1)
            else:
                window_reshaped = window
            
            # Get model predictions
            predictions = self.model(window_reshaped, training=False)
            Xhat_mean, Xhat_log_var = predictions[0], predictions[1]
            
            # Calculate reconstruction probability (unnormalized log prob)
            # This is consistent with the loss function during training
            std = tf.sqrt(tf.exp(Xhat_log_var))
            squared_error = tf.square(window_reshaped - Xhat_mean) / tf.square(std)
            
            # Sum over all elements and normalize by window size and features
            total_error = tf.reduce_sum(squared_error, axis=(-1, -2))
            n_elements = window_reshaped.shape[1] * window_reshaped.shape[2]  # time_steps * features
            anomaly_score = 0.5 * total_error / n_elements  # Normalize by number of elements
            
            return float(anomaly_score[0])
            
        except Exception as e:
            print(f"Error in calculate_anomaly_score: {str(e)}")
            return float('nan')
    
    def get_validation_scores_path(self) -> str:
        """Get the path for cached validation scores."""
        model_name = os.path.basename(os.path.normpath(self.model_path))
        return os.path.join(CACHE_DIR, f"validation_scores_{model_name}.npy")

    def set_threshold(self, validation_data: np.ndarray):
        """
        Set anomaly threshold using validation data.
        It will first try to load cached scores, otherwise it will compute and cache them.
        
        Args:
            validation_data: Pre-windowed validation data (shape: [n_windows, window_size, n_features])
        """
        print(f"\nCalculating threshold from validation data:")
        print(f"Validation data shape: {validation_data.shape}")

        scores_path = self.get_validation_scores_path()
        model_file_path = os.path.join(self.model_path, "saved_model.pb")
        
        try:
            # Get model's last modification time
            current_model_timestamp = os.path.getmtime(model_file_path)

            # Try to load cached scores
            cached_data = np.load(scores_path, allow_pickle=True).item()
            if (cached_data.get('model_path') == self.model_path and
                cached_data.get('model_timestamp') == current_model_timestamp):
                print(f"Loading cached validation scores from {scores_path}")
                scores = cached_data['scores']
            else:
                print("Model has been updated. Recalculating scores.")
                raise FileNotFoundError # Force recalculation
        except (FileNotFoundError, IOError, KeyError):
            # If cache doesn't exist, is invalid, or model file is missing, calculate scores
            print("Calculating validation scores...")
            scores = []
            total_windows = len(validation_data)
            
            # Process each window
            for i, window in enumerate(validation_data):
                if i % 100 == 0:
                    print(f"Processing window {i}/{total_windows}")
                    
                score = self.calculate_anomaly_score(window)
                
                if not np.isnan(score) and not np.isinf(score):
                    scores.append(score)
            
            # Get model's last modification time for caching
            try:
                current_model_timestamp = os.path.getmtime(model_file_path)
            except FileNotFoundError:
                print(f"Warning: Could not get timestamp for model at {model_file_path}. Cache may not be reliable.")
                current_model_timestamp = None

            # Cache the scores
            print(f"Caching validation scores to {scores_path}")
            np.save(scores_path, {
                'scores': np.array(scores), 
                'model_path': self.model_path,
                'model_timestamp': current_model_timestamp
            })
        
        if not isinstance(scores, list):
            scores = scores.tolist()

        if not scores:
            raise ValueError("No valid scores calculated from validation data")
        
        scores = np.array(scores)
        
        # Use robust statistics
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))  # Median Absolute Deviation
        
        # Set threshold as median + k * MAD
        k = 3.0  # Adjust this value based on desired sensitivity
        self.threshold = median + k * mad
        
        print(f"\nThreshold Statistics:")
        print(f"Median Score: {median:.4f}")
        print(f"MAD: {mad:.4f}")
        print(f"Threshold: {self.threshold:.4f}")
    
    def update_buffer(self, sample: np.ndarray) -> None:
        """
        Update the data buffer with a new sample
        
        Args:
            sample: New data sample
        """
        # Standardize sample
        sample_std = self.data_manager.standardize_sample(sample)
        
        # Add to buffer
        self.data_buffer.append(sample_std)
    
    def detect(self, sample: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if a new sample is anomalous
        
        Args:
            sample: New data sample
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Please call set_threshold() first.")
        
        # Update buffer with new sample
        self.update_buffer(sample)
        
        # If buffer is not full yet, return normal
        if len(self.data_buffer) < self.window_size:
            return False, 0.0
        
        # Convert buffer to numpy array
        window = np.array(self.data_buffer)
        
        # Calculate anomaly score
        score = self.calculate_anomaly_score(window)
        
        # Handle invalid scores
        if np.isnan(score) or np.isinf(score):
            return False, score
        
        # Use exponential moving average for smoothing
        if not hasattr(self, 'smoothed_score'):
            self.smoothed_score = score
        else:
            alpha = 0.1  # Smoothing factor
            self.smoothed_score = alpha * score + (1 - alpha) * self.smoothed_score
        
        # Determine if anomalous
        is_anomaly = self.smoothed_score > self.threshold
        
        return is_anomaly, self.smoothed_score

    def set_threshold_from_raw_data(self, normal_data: List[np.ndarray]):
        """
        Set anomaly threshold using raw data processed the same way as detection
        
        Args:
            normal_data: List of raw normal sequences (same format as detection input)
        """
        print(f"\nCalculating threshold from raw normal data:")
        print(f"Number of normal sequences: {len(normal_data)}")
        
        scores = []
        
        # Process each normal sequence exactly like detection
        for seq_idx, sequence in enumerate(normal_data):
            print(f"Processing normal sequence {seq_idx + 1}/{len(normal_data)}")
            
            # Reset buffer for each sequence
            self.data_buffer.clear()
            
            # Process each sample in the sequence
            for sample in sequence:
                # Update buffer with new sample (same as detection)
                self.update_buffer(sample)
                
                # Calculate score only when buffer is full
                if len(self.data_buffer) == self.window_size:
                    window = np.array(self.data_buffer)
                    score = self.calculate_anomaly_score(window)
                    
                    if not np.isnan(score) and not np.isinf(score):
                        scores.append(score)
        
        if not scores:
            raise ValueError("No valid scores calculated from normal data")
        
        scores = np.array(scores)
        
        # Use robust statistics
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))  # Median Absolute Deviation
        
        # Set threshold as median + k * MAD
        k = 3.0  # Adjust this value based on desired sensitivity
        self.threshold = median + k * mad
        
        print(f"\nThreshold Statistics (from raw data):")
        print(f"Total valid scores: {len(scores)}")
        print(f"Median Score: {median:.4f}")
        print(f"MAD: {mad:.4f}")
        print(f"Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        print(f"Threshold: {self.threshold:.4f}")

class StreamingDetector:
    def __init__(self,
                 model_path: str,
                 output_dir: str = "output"):
        """
        Initialize streaming detector for real-time anomaly detection
        
        Args:
            model_path: Path to saved MA-VAE model
            output_dir: Directory containing processed data
        """
        print("\nInitializing StreamingDetector...")
        print(f"Model path: {model_path}")
        print(f"Output directory: {output_dir}")
        
        # Initialize data manager and load data
        self.data_manager = DataManager(output_dir)
        try:
            self.data_manager.load_scaler()
        except FileNotFoundError as e:
            raise ValueError(f"Could not load scaler. Please train model first: {str(e)}")
        
        # Initialize detector
        self.detector = MAVAEDetector(model_path, self.data_manager)
        
        # Load normal data and set threshold using raw data processing
        try:
            print("\nLoading raw normal data for threshold setting...")
            # Load raw normal data (same as used for detection)
            normal_data, _ = self.data_manager.load_mat_data("data.mat")
            
            # Set threshold using raw data processed the same way as detection
            self.detector.set_threshold_from_raw_data(normal_data)
            
        except Exception as e:
            raise ValueError(f"Error processing normal data: {str(e)}")
        
        self.anomaly_history = []
        self.score_history = []
    
    def process_sample(self, sample: np.ndarray) -> Dict[str, float]:
        """
        Process a new sample in real-time
        
        Args:
            sample: New data sample
            
        Returns:
            Dictionary containing detection results
        """
        # Detect anomaly
        is_anomaly, score = self.detector.detect(sample)
        
        # Update history
        self.anomaly_history.append(is_anomaly)
        self.score_history.append(score)
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': score,
            'threshold': self.detector.threshold
        }
    
    def get_detection_stats(self) -> Dict[str, float]:
        """Get current detection statistics"""
        if not self.anomaly_history:
            return {}
        
        valid_scores = [s for s in self.score_history if not np.isnan(s) and not np.isinf(s)]
        
        return {
            'total_samples': len(self.anomaly_history),
            'anomaly_rate': sum(self.anomaly_history) / len(self.anomaly_history),
            'current_score': self.score_history[-1] if self.score_history else None,
            'mean_score': np.mean(valid_scores) if valid_scores else None,
            'threshold': self.detector.threshold
        }

if __name__ == "__main__":
    # Initialize detector
    detector = StreamingDetector(
        model_path="output/model",
        output_dir="output"
    )
    
    # Load raw test data to simulate a streaming scenario
    print("\nLoading raw test data...")
    try:
        normal_data, abnormal_data = detector.data_manager.load_mat_data("data.mat")
    except FileNotFoundError:
        print("Error: data.mat not found. Please ensure it is in the root directory.")
        exit(1)
        
    print(f"Loaded {len(normal_data)} normal sequences for testing")
    print(f"Loaded {len(abnormal_data)} abnormal sequences for testing")
    
    # Process normal sequences
    print("\nProcessing normal sequences...")
    normal_results = []
    for seq_idx, sequence in enumerate(normal_data):
        print(f"\nProcessing normal sequence {seq_idx + 1}/{len(normal_data)}")
        
        # Process each sample in the sequence
        detection_points = []
        sequence_scores = []
        for t, sample in enumerate(sequence):
            result = detector.process_sample(sample)
            sequence_scores.append(result['anomaly_score'])
            
            if result['is_anomaly']:
                detection_points.append(t)
                print(f"  False alarm at timestep {t}")
                print(f"  Score: {result['anomaly_score']:.2f}")
                print(f"  Threshold: {result['threshold']:.2f}")
        
        # Store sequence results
        normal_results.append({
            'sequence_idx': seq_idx,
            'detection_points': detection_points,
            'scores': sequence_scores,
            'false_alarm_rate': len(detection_points) / len(sequence) if len(sequence) > 0 else 0
        })
        
        # Print sequence summary
        if detection_points:
            print(f"\nNormal Sequence {seq_idx + 1} Summary:")
            print(f"  False alarms: {len(detection_points)}")
            print(f"  False alarm rate: {normal_results[-1]['false_alarm_rate']:.2%}")
    
    # Process abnormal sequences
    print("\nProcessing abnormal sequences...")
    abnormal_results = []
    for seq_idx, sequence in enumerate(abnormal_data):
        print(f"\nProcessing abnormal sequence {seq_idx + 1}/{len(abnormal_data)}")
        
        # Process each sample in the sequence
        detection_points = []
        sequence_scores = []
        for t, sample in enumerate(sequence):
            result = detector.process_sample(sample)
            sequence_scores.append(result['anomaly_score'])
            
            if result['is_anomaly']:
                detection_points.append(t)
                print(f"  Anomaly detected at timestep {t}")
                print(f"  Score: {result['anomaly_score']:.2f}")
                print(f"  Threshold: {result['threshold']:.2f}")
        
        # Store sequence results
        abnormal_results.append({
            'sequence_idx': seq_idx + len(normal_data),  # Global index
            'detection_points': detection_points,
            'scores': sequence_scores,
            'detection_rate': len(detection_points) / len(sequence) if len(sequence) > 0 else 0,
            'detection_delay': min(detection_points) if detection_points else float('inf')
        })
        
        # Print sequence summary
        if detection_points:
            print(f"\nAbnormal Sequence {seq_idx + 1} Summary:")
            print(f"  First detection at timestep {abnormal_results[-1]['detection_delay']}")
            print(f"  Total detections: {len(detection_points)}")
            print(f"  Detection rate: {abnormal_results[-1]['detection_rate']:.2%}")
        else:
            print(f"\nNo anomalies detected in abnormal sequence {seq_idx + 1}")
    
    # Print overall statistics
    print("\n=== Overall Detection Statistics ===")
    
    # Normal sequence statistics
    total_false_alarms = sum(len(r['detection_points']) for r in normal_results)
    avg_false_alarm_rate = np.mean([r['false_alarm_rate'] for r in normal_results])
    print("\nNormal Sequences:")
    print(f"Total false alarms: {total_false_alarms}")
    print(f"Average false alarm rate: {avg_false_alarm_rate:.2%}")
    
    # Abnormal sequence statistics
    detected_sequences = sum(1 for r in abnormal_results if r['detection_points'])
    avg_detection_rate = np.mean([r['detection_rate'] for r in abnormal_results])
    avg_detection_delay = np.mean([r['detection_delay'] for r in abnormal_results if r['detection_points']])
    print("\nAbnormal Sequences:")
    print(f"Detected sequences: {detected_sequences}/{len(abnormal_data)}")
    print(f"Average detection rate: {avg_detection_rate:.2%}")
    print(f"Average detection delay: {avg_detection_delay:.2f} timesteps")
    
    # Prepare results
    results = {
        'normal_results': [
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()}
            for r in normal_results
        ],
        'abnormal_results': [
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()}
            for r in abnormal_results
        ],
        'statistics': {
            'total_false_alarms': total_false_alarms,
            'avg_false_alarm_rate': float(avg_false_alarm_rate),
            'detected_sequences': detected_sequences,
            'detection_rate': float(detected_sequences / len(abnormal_data)),
            'avg_detection_rate': float(avg_detection_rate),
            'avg_detection_delay': float(avg_detection_delay)
        }
    }
    
    # Save results in multiple formats
    output_dir = "output"
    
    # Save as JSON (human-readable)
    with open(os.path.join(output_dir, "detection_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as numpy array (for later processing)
    np.save(os.path.join(output_dir, "detection_results.npy"), results)
    
    # Generate detailed report
    generate_detection_report(results, output_dir)
    print(f"\nDetailed report saved to: {os.path.join(output_dir, 'detection_report.txt')}") 