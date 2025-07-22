import numpy as np
from detect import StreamingDetector, MAVAEDetector
from data_manager import DataManager
import matplotlib.pyplot as plt

def run_detection_diagnostics():
    """
    Run diagnostics on the detection system to compare threshold calculation methods
    """
    print("=== MA-VAE Detection Diagnostics ===\n")
    
    # Initialize detector
    try:
        detector = StreamingDetector(
            model_path="output/model",
            output_dir="output"
        )
        print("✓ Detection system initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize detection system: {e}")
        return
    
    # Load test data
    try:
        normal_data, abnormal_data = detector.data_manager.load_mat_data("data.mat")
        print(f"✓ Loaded {len(normal_data)} normal and {len(abnormal_data)} abnormal sequences")
    except Exception as e:
        print(f"✗ Failed to load test data: {e}")
        return
    
    # Test a few normal sequences
    print("\n=== Testing Normal Sequences ===")
    normal_results = []
    
    for seq_idx in range(min(3, len(normal_data))):  # Test first 3 sequences
        sequence = normal_data[seq_idx]
        print(f"\nProcessing normal sequence {seq_idx + 1} (length: {len(sequence)}):")
        
        # Reset detector state for each sequence
        detector.detector.data_buffer.clear()
        if hasattr(detector.detector, 'smoothed_score'):
            delattr(detector.detector, 'smoothed_score')
        
        sequence_scores = []
        detection_points = []
        
        # Process each sample
        for t, sample in enumerate(sequence):
            result = detector.process_sample(sample)
            sequence_scores.append(result['anomaly_score'])
            
            if result['is_anomaly']:
                detection_points.append(t)
        
        # Calculate statistics
        valid_scores = [s for s in sequence_scores if s > 0]  # Exclude 0.0 scores from buffer filling
        
        print(f"  - Valid scores: {len(valid_scores)}")
        print(f"  - Score range: [{min(valid_scores):.4f}, {max(valid_scores):.4f}]")
        print(f"  - Mean score: {np.mean(valid_scores):.4f}")
        print(f"  - Threshold: {detector.detector.threshold:.4f}")
        print(f"  - False alarms: {len(detection_points)} at timesteps {detection_points[:10]}...")
        print(f"  - False alarm rate: {len(detection_points)/len(sequence):.2%}")
        
        normal_results.append({
            'sequence_idx': seq_idx,
            'scores': sequence_scores,
            'detection_points': detection_points,
            'false_alarm_rate': len(detection_points)/len(sequence)
        })
    
    # Test a few abnormal sequences
    print("\n=== Testing Abnormal Sequences ===")
    abnormal_results = []
    
    for seq_idx in range(min(2, len(abnormal_data))):  # Test first 2 sequences
        sequence = abnormal_data[seq_idx]
        print(f"\nProcessing abnormal sequence {seq_idx + 1} (length: {len(sequence)}):")
        
        # Reset detector state for each sequence
        detector.detector.data_buffer.clear()
        if hasattr(detector.detector, 'smoothed_score'):
            delattr(detector.detector, 'smoothed_score')
        
        sequence_scores = []
        detection_points = []
        
        # Process each sample
        for t, sample in enumerate(sequence):
            result = detector.process_sample(sample)
            sequence_scores.append(result['anomaly_score'])
            
            if result['is_anomaly']:
                detection_points.append(t)
        
        # Calculate statistics
        valid_scores = [s for s in sequence_scores if s > 0]
        
        print(f"  - Valid scores: {len(valid_scores)}")
        print(f"  - Score range: [{min(valid_scores):.4f}, {max(valid_scores):.4f}]")
        print(f"  - Mean score: {np.mean(valid_scores):.4f}")
        print(f"  - Threshold: {detector.detector.threshold:.4f}")
        print(f"  - Detections: {len(detection_points)} at timesteps {detection_points[:10]}...")
        print(f"  - Detection rate: {len(detection_points)/len(sequence):.2%}")
        print(f"  - First detection at: {min(detection_points) if detection_points else 'N/A'}")
        
        abnormal_results.append({
            'sequence_idx': seq_idx,
            'scores': sequence_scores,
            'detection_points': detection_points,
            'detection_rate': len(detection_points)/len(sequence)
        })
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    avg_false_alarm_rate = np.mean([r['false_alarm_rate'] for r in normal_results])
    avg_detection_rate = np.mean([r['detection_rate'] for r in abnormal_results])
    
    print(f"Average false alarm rate: {avg_false_alarm_rate:.2%}")
    print(f"Average detection rate: {avg_detection_rate:.2%}")
    print(f"Threshold value: {detector.detector.threshold:.4f}")
    
    # Plot score distributions
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Score evolution for first normal sequence
    plt.subplot(1, 3, 1)
    if normal_results:
        scores = normal_results[0]['scores']
        plt.plot(scores, 'b-', label='Normal Sequence 1')
        plt.axhline(y=detector.detector.threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Timestep')
        plt.ylabel('Anomaly Score')
        plt.title('Score Evolution - Normal Sequence')
        plt.legend()
        plt.yscale('log')
    
    # Plot 2: Score evolution for first abnormal sequence
    plt.subplot(1, 3, 2)
    if abnormal_results:
        scores = abnormal_results[0]['scores']
        plt.plot(scores, 'r-', label='Abnormal Sequence 1')
        plt.axhline(y=detector.detector.threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Timestep')
        plt.ylabel('Anomaly Score')
        plt.title('Score Evolution - Abnormal Sequence')
        plt.legend()
        plt.yscale('log')
    
    # Plot 3: Score distribution
    plt.subplot(1, 3, 3)
    all_normal_scores = []
    all_abnormal_scores = []
    
    for r in normal_results:
        all_normal_scores.extend([s for s in r['scores'] if s > 0])
    
    for r in abnormal_results:
        all_abnormal_scores.extend([s for s in r['scores'] if s > 0])
    
    if all_normal_scores:
        plt.hist(all_normal_scores, bins=30, alpha=0.5, label='Normal', density=True, color='blue')
    if all_abnormal_scores:
        plt.hist(all_abnormal_scores, bins=30, alpha=0.5, label='Abnormal', density=True, color='red')
    
    plt.axvline(x=detector.detector.threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('output/detection_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Diagnostic plots saved to output/detection_diagnostics.png")

if __name__ == "__main__":
    run_detection_diagnostics() 