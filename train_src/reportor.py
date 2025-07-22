from datetime import datetime
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional, Dict
from config import *
import os

def generate_detection_report(results: Dict, output_dir: str):
    """Generate a detailed detection report in human-readable format"""
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(output_dir, "detection_report.txt")
    
    with open(report_path, 'w') as f:
        # Header
        f.write("=== MA-VAE Anomaly Detection Report ===\n")
        f.write(f"Generated at: {report_time}\n\n")
        
        # Model Configuration
        f.write("Model Configuration:\n")
        f.write(f"- Window Size: {WINDOW_SIZE}\n")
        f.write(f"- Window Shift: {WINDOW_SHIFT}\n")
        f.write(f"- Features: {N_FEATURES}\n")
        f.write(f"- LSTM Units: {LSTM_UNITS}\n\n")
        
        # Overall Statistics
        stats = results['statistics']
        f.write("Overall Detection Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write("\nNormal Sequence Analysis:\n")
        f.write(f"- Total False Alarms: {stats['total_false_alarms']}\n")
        f.write(f"- Average False Alarm Rate: {stats['avg_false_alarm_rate']:.2%}\n")
        
        f.write("\nAbnormal Sequence Analysis:\n")
        f.write(f"- Detected Sequences: {stats['detected_sequences']}/{len(results['abnormal_results'])}\n")
        f.write(f"- Detection Rate: {stats['detection_rate']:.2%}\n")
        f.write(f"- Average Detection Rate: {stats['avg_detection_rate']:.2%}\n")
        f.write(f"- Average Detection Delay: {stats['avg_detection_delay']:.2f} timesteps\n\n")
        
        # Detailed Normal Sequence Results
        f.write("\nDetailed Normal Sequence Results:\n")
        f.write("-" * 30 + "\n")
        for idx, result in enumerate(results['normal_results']):
            if result['detection_points']:
                f.write(f"\nSequence {idx + 1}:\n")
                f.write(f"- False Alarms at timesteps: {result['detection_points']}\n")
                f.write(f"- False Alarm Rate: {result['false_alarm_rate']:.2%}\n")
        
        # Detailed Abnormal Sequence Results
        f.write("\nDetailed Abnormal Sequence Results:\n")
        f.write("-" * 30 + "\n")
        for idx, result in enumerate(results['abnormal_results']):
            f.write(f"\nSequence {idx + 1}:\n")
            if result['detection_points']:
                f.write(f"- First Detection at: timestep {result['detection_delay']}\n")
                f.write(f"- All Detections at timesteps: {result['detection_points']}\n")
                f.write(f"- Detection Rate: {result['detection_rate']:.2%}\n")
            else:
                f.write("- No anomalies detected\n")
        
        f.write("\n=== End of Report ===\n")
