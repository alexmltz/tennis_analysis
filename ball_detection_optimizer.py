"""
Ball Detection Optimizer

This script tests different confidence thresholds to optimize ball detection rate.
Goal: Achieve >80% detection rate in raw detections (before interpolation).
"""

from utils import read_video_limited, get_video_info
from trackers import BallTracker
import pandas as pd
import numpy as np


def analyze_ball_detection_with_confidence(frames, model_path, confidence_threshold):
    """
    Test ball detection with a specific confidence threshold
    """
    # Create a custom tracker for this test
    from ultralytics import YOLO
    import pickle
    
    class TestBallTracker:
        def __init__(self, model_path):
            self.model = YOLO(model_path)
        
        def detect_frame(self, frame, conf_threshold):
            results = self.model.predict(frame, conf=conf_threshold)[0]
            ball_dict = {}
            for box in results.boxes:
                result = box.xyxy.tolist()[0]
                ball_dict[1] = result
                break  # Take only the first detection (highest confidence)
            return ball_dict
        
        def detect_frames(self, frames, conf_threshold):
            ball_detections = []
            for frame in frames:
                ball_dict = self.detect_frame(frame, conf_threshold)
                ball_detections.append(ball_dict)
            return ball_detections
    
    tracker = TestBallTracker(model_path)
    detections = tracker.detect_frames(frames, confidence_threshold)
    
    # Count detections
    frames_with_ball = sum(1 for detection in detections if len(detection) > 0 and 1 in detection)
    total_frames = len(detections)
    detection_rate = frames_with_ball / total_frames * 100
    
    return detection_rate, frames_with_ball, total_frames


def main():
    print("🎾 BALL DETECTION OPTIMIZER")
    print("="*50)
    
    # Configuration
    input_video_path = "input_videos/input_video.mp4"
    max_frames = 250
    model_path = 'models/yolo5_last.pt'
    
    # Read frames
    print(f"Reading first {max_frames} frames...")
    video_frames = read_video_limited(input_video_path, max_frames=max_frames, start_frame=0)
    print(f"Loaded {len(video_frames)} frames")
    
    # Test different confidence thresholds
    confidence_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    print(f"\nTesting confidence thresholds...")
    print("Confidence | Detection Rate | Frames Detected | Total Frames")
    print("-" * 60)
    
    results = []
    best_conf = None
    best_rate = 0
    
    for conf in confidence_thresholds:
        try:
            detection_rate, frames_detected, total_frames = analyze_ball_detection_with_confidence(
                video_frames, model_path, conf
            )
            
            results.append({
                'confidence': conf,
                'detection_rate': detection_rate,
                'frames_detected': frames_detected,
                'total_frames': total_frames
            })
            
            status = "✅" if detection_rate >= 80 else "⚠️" if detection_rate >= 60 else "❌"
            print(f"   {conf:.2f}    |     {detection_rate:5.1f}%     |      {frames_detected:3d}        |     {total_frames}")
            
            if detection_rate >= 80 and (best_conf is None or conf > best_conf):
                best_conf = conf
                best_rate = detection_rate
                
        except Exception as e:
            print(f"   {conf:.2f}    |     ERROR      |       -         |     -")
            print(f"           Error: {str(e)}")
    
    print("-" * 60)
    
    # Analysis and recommendations
    print(f"\n🎯 OPTIMIZATION RESULTS:")
    
    if best_conf is not None:
        print(f"✅ SUCCESS: Found optimal confidence threshold!")
        print(f"   Best confidence: {best_conf:.2f}")
        print(f"   Detection rate: {best_rate:.1f}%")
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Update ball_tracker.py line 75:")
        print(f"   results = self.model.predict(frame,conf={best_conf})[0]")
    else:
        # Find the highest rate below 80%
        valid_results = [r for r in results if r['detection_rate'] > 0]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['detection_rate'])
            print(f"❌ Target not achieved, but best result:")
            print(f"   Best confidence: {best_result['confidence']:.2f}")
            print(f"   Detection rate: {best_result['detection_rate']:.1f}%")
            print(f"   Need: {80 - best_result['detection_rate']:.1f} more percentage points")
            
            print(f"\n💡 ALTERNATIVE OPTIMIZATIONS:")
            print(f"   1. Try even lower confidence thresholds (0.01-0.05)")
            print(f"   2. Check if model file exists and is correct format")
            print(f"   3. Try different YOLO model versions")
            print(f"   4. Improve training data quality")
        else:
            print(f"❌ No valid detections found at any confidence level")
            print(f"💡 Check model path and format: {model_path}")
    
    print(f"\n📊 DETAILED RESULTS:")
    for result in results:
        target_met = "🎯" if result['detection_rate'] >= 80 else "  "
        print(f"   {target_met} Conf {result['confidence']:.2f}: {result['detection_rate']:5.1f}% ({result['frames_detected']}/{result['total_frames']} frames)")


if __name__ == "__main__":
    main() 