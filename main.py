"""
Tennis Analysis Main Script - Ball Detection Focus

This version focuses on ball detection optimization for the first 250 frames.
Tracks and reports ball detection rate to optimize until >80% detection is achieved.
"""

from utils import (read_video, 
                   save_video,
                   read_video_limited,
                   read_video_sampled,
                   get_video_info,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy


def analyze_ball_detection_quality(ball_detections, total_frames):
    """
    Analyze ball detection quality and return statistics
    """
    frames_with_ball = 0
    frames_without_ball = 0
    detection_details = []
    
    for frame_idx, ball_dict in enumerate(ball_detections):
        has_ball = len(ball_dict) > 0 and 1 in ball_dict
        if has_ball:
            frames_with_ball += 1
            bbox = ball_dict[1]
            detection_details.append({
                'frame': frame_idx,
                'detected': True,
                'bbox': bbox
            })
        else:
            frames_without_ball += 1
            detection_details.append({
                'frame': frame_idx,
                'detected': False,
                'bbox': None
            })
    
    detection_rate = frames_with_ball / total_frames * 100
    
    return {
        'total_frames': total_frames,
        'frames_with_ball': frames_with_ball,
        'frames_without_ball': frames_without_ball,
        'detection_rate': detection_rate,
        'details': detection_details
    }


def main():
    try:
        print("Starting tennis analysis - BALL DETECTION FOCUS...")
        print("="*60)
        
        # Configuration: Process first 250 frames only
        input_video_path = "input_videos/input_video.mp4"
        max_frames = 250
        
        # Get video info
        video_info = get_video_info(input_video_path)
        print(f"Video info: {video_info}")
        print(f"Processing first {max_frames} frames for ball detection optimization...")
        
        # Read limited frames
        video_frames = read_video_limited(input_video_path, max_frames=max_frames, start_frame=0)
        print(f"Successfully read {len(video_frames)} frames")
        
        # Initialize trackers
        print("\nInitializing trackers...")
        player_tracker = PlayerTracker(model_path='yolov8x')
        ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
        print("Trackers initialized successfully")

        # Ball Detection - MAIN FOCUS
        print("\n" + "="*50)
        print("🎾 BALL DETECTION ANALYSIS")
        print("="*50)
        
        print("Detecting ball in all frames...")
        ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/ball_detections.pkl")
        print("Ball detection completed")
        
        # Analyze ball detection quality
        ball_stats = analyze_ball_detection_quality(ball_detections, len(video_frames))
        
        print(f"\n🎯 BALL DETECTION RESULTS:")
        print(f"  Total frames analyzed: {ball_stats['total_frames']}")
        print(f"  Frames with ball detected: {ball_stats['frames_with_ball']}")
        print(f"  Frames without ball: {ball_stats['frames_without_ball']}")
        print(f"  Detection rate: {ball_stats['detection_rate']:.1f}%")
        
        if ball_stats['detection_rate'] >= 80:
            print(f"  ✅ SUCCESS: Detection rate {ball_stats['detection_rate']:.1f}% >= 80% target!")
        elif ball_stats['detection_rate'] >= 60:
            print(f"  ⚠️  GOOD: Detection rate {ball_stats['detection_rate']:.1f}% is good but not optimal")
        elif ball_stats['detection_rate'] >= 40:
            print(f"  🔶 MODERATE: Detection rate {ball_stats['detection_rate']:.1f}% needs improvement")
        else:
            print(f"  ❌ POOR: Detection rate {ball_stats['detection_rate']:.1f}% needs significant improvement")
        
        # Show detection patterns
        print(f"\n📊 Detection Pattern Analysis:")
        consecutive_misses = 0
        max_consecutive_misses = 0
        current_streak = 0
        
        for i, detection in enumerate(ball_stats['details']):
            if detection['detected']:
                if consecutive_misses > 0:
                    max_consecutive_misses = max(max_consecutive_misses, consecutive_misses)
                    consecutive_misses = 0
                current_streak += 1
            else:
                consecutive_misses += 1
                current_streak = 0
            
            # Show first 10 and last 10 frames as examples
            if i < 10 or i >= len(ball_stats['details']) - 10:
                status = "✅" if detection['detected'] else "❌"
                print(f"  Frame {i:3d}: {status}")
        
        print(f"\n  Max consecutive misses: {max_consecutive_misses}")
        
        # Interpolate ball positions
        print("\nInterpolating ball positions...")
        ball_detections_interpolated = ball_tracker.interpolate_ball_positions(ball_detections)
        
        # Analyze interpolated results
        interpolated_stats = analyze_ball_detection_quality(ball_detections_interpolated, len(video_frames))
        print(f"\n🔄 AFTER INTERPOLATION:")
        print(f"  Detection rate: {interpolated_stats['detection_rate']:.1f}%")
        print(f"  Improvement: +{interpolated_stats['detection_rate'] - ball_stats['detection_rate']:.1f} percentage points")
        
        # Quick player detection (minimal processing)
        print("\n" + "="*30)
        print("QUICK PLAYER DETECTION")
        print("="*30)
        
        print("Detecting players...")
        player_detections = player_tracker.detect_frames(video_frames,
                                                         read_from_stub=False,
                                                         stub_path="tracker_stubs/player_detections.pkl")
        
        # Court Line Detector
        print("Predicting court keypoints...")
        court_model_path = "models/keypoints_model.pth"
        court_line_detector = CourtLineDetector(court_model_path)
        court_keypoints = court_line_detector.predict(video_frames[0])
        
        # Filter players
        player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
        
        # Quick visualization output
        print("\n" + "="*30)
        print("CREATING OUTPUT VIDEO")
        print("="*30)
        
        print("Drawing bounding boxes...")
        output_video_frames = ball_tracker.draw_bboxes(video_frames.copy(), ball_detections_interpolated)
        output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections)
        output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
        
        # Add detection rate overlay on each frame
        for i, frame in enumerate(output_video_frames):
            # Frame number
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Ball detection status for this frame
            ball_detected = "YES" if ball_stats['details'][i]['detected'] else "NO"
            color = (0, 255, 0) if ball_stats['details'][i]['detected'] else (0, 0, 255)
            cv2.putText(frame, f"Ball: {ball_detected}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Overall detection rate
            cv2.putText(frame, f"Detection Rate: {ball_stats['detection_rate']:.1f}%", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Target status
            target_status = "TARGET MET!" if ball_stats['detection_rate'] >= 80 else "OPTIMIZE MORE"
            target_color = (0, 255, 0) if ball_stats['detection_rate'] >= 80 else (0, 165, 255)
            cv2.putText(frame, target_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, target_color, 2)
        
        print("Saving output video...")
        save_video(output_video_frames, "output_videos/output_video.avi")
        
        # Final summary
        print("\n" + "="*60)
        print("🏆 FINAL BALL DETECTION SUMMARY")
        print("="*60)
        print(f"Frames processed: {len(video_frames)}")
        print(f"Ball detection rate: {ball_stats['detection_rate']:.1f}%")
        print(f"After interpolation: {interpolated_stats['detection_rate']:.1f}%")
        
        if interpolated_stats['detection_rate'] >= 80:
            print("✅ SUCCESS: Target of >80% detection rate achieved!")
        else:
            print(f"❌ TARGET NOT MET: Need {80 - interpolated_stats['detection_rate']:.1f} more percentage points")
            print("\n💡 OPTIMIZATION SUGGESTIONS:")
            print("  1. Adjust confidence threshold in ball_tracker.py (currently 0.15)")
            print("  2. Try different YOLO model versions")
            print("  3. Improve training data or model fine-tuning")
            print("  4. Check if ball detection model path is correct")
        
        print("\nAnalysis completed!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()