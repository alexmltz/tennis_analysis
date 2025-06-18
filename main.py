"""
Tennis Analysis Main Script

DEBUGGING SUMMARY:
- Fixed memory issues by implementing limited frame processing (read_video_limited)
- Fixed KeyError in mini_court by handling unknown player IDs gracefully
- Fixed pandas deprecation warnings in ball_tracker.py
- Fixed player statistics calculation by mapping arbitrary player IDs to 1,2
- Fixed index out of range in player stats drawing by adding bounds checking
- Fixed PyTorch deprecation warning in court_line_detector.py

The script now runs successfully on limited frames. To process full video, 
increase max_frames or set to None (but may require more memory).
"""

from utils import (read_video, 
                   save_video,
                   read_video_limited,
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


def main():
    try:
        print("Starting tennis analysis...")
        
        # Get video info first
        input_video_path = "input_videos/input_video.mp4"
        video_info = get_video_info(input_video_path)
        print(f"Video info: {video_info}")
        
        # Configuration: Set max_frames to None to process full video (requires more memory)
        max_frames = 500  # Process more frames to get ball shots and better analysis
        # max_frames = None  # Uncomment this line to process full video
        
        if max_frames:
            print(f"Reading first {max_frames} frames from: {input_video_path}")
            video_frames = read_video_limited(input_video_path, max_frames=max_frames)
        else:
            print(f"Reading all frames from: {input_video_path}")
            video_frames = read_video(input_video_path)
            
        print(f"Successfully read {len(video_frames)} frames")

        # Detect Players and Ball
        print("Initializing trackers...")
        player_tracker = PlayerTracker(model_path='yolov8x')
        ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
        print("Trackers initialized successfully")

        print("Detecting players...")
        player_detections = player_tracker.detect_frames(video_frames,
                                                         read_from_stub=False,
                                                         stub_path="tracker_stubs/player_detections.pkl"
                                                         )
        print("Player detection completed")
        
        print("Detecting ball...")
        ball_detections = ball_tracker.detect_frames(video_frames,
                                                         read_from_stub=False,
                                                         stub_path="tracker_stubs/ball_detections.pkl"
                                                         )
        print("Ball detection completed")
        
        print("Interpolating ball positions...")
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
        print("Ball interpolation completed")
        
        # Court Line Detector model
        print("Initializing court line detector...")
        court_model_path = "models/keypoints_model.pth"
        court_line_detector = CourtLineDetector(court_model_path)
        print("Predicting court keypoints...")
        court_keypoints = court_line_detector.predict(video_frames[0])
        print("Court keypoints prediction completed")

        # choose players
        print("Choosing and filtering players...")
        player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
        print("Player filtering completed")

        # MiniCourt
        print("Initializing mini court...")
        mini_court = MiniCourt(video_frames[0]) 
        print("Mini court initialized")

        # Detect ball shots
        print("Detecting ball shots...")
        ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)
        print(f"Found {len(ball_shot_frames)} ball shot frames")

        # Convert positions to mini court positions
        print("Converting positions to mini court coordinates...")
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                              ball_detections,
                                                                                                              court_keypoints)
        print("Position conversion completed")

        print("Calculating player statistics...")
        player_stats_data = [{
            'frame_num':0,
            'player_1_number_of_shots':0,
            'player_1_total_shot_speed':0,
            'player_1_last_shot_speed':0,
            'player_1_total_player_speed':0,
            'player_1_last_player_speed':0,

            'player_2_number_of_shots':0,
            'player_2_total_shot_speed':0,
            'player_2_last_shot_speed':0,
            'player_2_total_player_speed':0,
            'player_2_last_player_speed':0,
        } ]
        
        for ball_shot_ind in range(len(ball_shot_frames)-1):
            start_frame = ball_shot_frames[ball_shot_ind]
            end_frame = ball_shot_frames[ball_shot_ind+1]
            
            # Skip if frames are beyond our limited frame range
            if start_frame >= len(video_frames) or end_frame >= len(video_frames):
                continue
            
            # Skip if we don't have player data for these frames
            if (start_frame >= len(player_mini_court_detections) or 
                end_frame >= len(player_mini_court_detections) or
                len(player_mini_court_detections[start_frame]) == 0 or
                len(player_mini_court_detections[end_frame]) == 0):
                print(f"Skipping frames {start_frame}-{end_frame}: insufficient player data")
                continue
                
            ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

            # Get distance covered by the ball
            distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                               ball_mini_court_detections[end_frame][1])
            distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court()
                                                                               ) 

            # Speed of the ball shot in km/h
            speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

            # player who the ball
            player_positions = player_mini_court_detections[start_frame]
            if not player_positions:
                print(f"Skipping frame {start_frame}: no player positions")
                continue
                
            player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                     ball_mini_court_detections[start_frame][1]))

            # opponent player speed - need to handle case where we don't have both players
            available_players = list(player_positions.keys())
            if len(available_players) < 2:
                print(f"Skipping frame {start_frame}: only {len(available_players)} players detected")
                continue
            
            # Find the opponent player ID
            opponent_player_id = None
            for pid in available_players:
                if pid != player_shot_ball:
                    opponent_player_id = pid
                    break
            
            if opponent_player_id is None:
                print(f"Skipping frame {start_frame}: could not find opponent player")
                continue
            
            # Check if opponent exists in both frames
            if (opponent_player_id not in player_mini_court_detections[start_frame] or
                opponent_player_id not in player_mini_court_detections[end_frame]):
                print(f"Skipping frame {start_frame}: opponent player {opponent_player_id} not in both frames")
                continue
            
            distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                    player_mini_court_detections[end_frame][opponent_player_id])
            distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court()
                                                                               ) 

            speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

            # Map player IDs to 1 and 2 for statistics (since our stats structure expects player_1 and player_2)
            available_player_ids = sorted(list(player_positions.keys()))
            player_id_mapping = {available_player_ids[i]: i+1 for i in range(min(2, len(available_player_ids)))}
            
            mapped_player_shot_ball = player_id_mapping.get(player_shot_ball, 1)
            mapped_opponent_player_id = player_id_mapping.get(opponent_player_id, 2)
            
            current_player_stats= deepcopy(player_stats_data[-1])
            current_player_stats['frame_num'] = start_frame
            current_player_stats[f'player_{mapped_player_shot_ball}_number_of_shots'] += 1
            current_player_stats[f'player_{mapped_player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
            current_player_stats[f'player_{mapped_player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

            current_player_stats[f'player_{mapped_opponent_player_id}_total_player_speed'] += speed_of_opponent
            current_player_stats[f'player_{mapped_opponent_player_id}_last_player_speed'] = speed_of_opponent

            player_stats_data.append(current_player_stats)

        print("Creating player statistics dataframe...")
        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
        player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
        player_stats_data_df = player_stats_data_df.ffill()

        # Fix the average speed calculations - there were bugs in the original code
        player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots'].replace(0, 1)
        player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots'].replace(0, 1)
        player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_1_number_of_shots'].replace(0, 1)
        player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_2_number_of_shots'].replace(0, 1)

        print("Drawing output frames...")
        # Draw output
        ## Draw Player Bounding Boxes
        output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
        output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

        ## Draw court Keypoints
        output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

        # Draw Mini Court
        output_video_frames = mini_court.draw_mini_court(output_video_frames)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

        # Draw Player Stats
        output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

        ## Draw frame number on top left corner
        for i, frame in enumerate(output_video_frames):
            cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        print("Saving output video...")
        save_video(output_video_frames, "output_videos/output_video.avi")
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()