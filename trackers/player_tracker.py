from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox, get_foot_position

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        if len(player_dict) == 0:
            return []
        if len(player_dict) == 1:
            return list(player_dict.keys())
        if len(player_dict) == 2:
            # Perfect case - exactly 2 players detected
            return list(player_dict.keys())
        
        # More than 2 players - need to be more selective
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            player_foot = get_foot_position(bbox)  # Use foot position as it's more relevant for court proximity

            # Calculate distance to court center and court lines
            min_distance_to_keypoints = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_foot, court_keypoint)
                if distance < min_distance_to_keypoints:
                    min_distance_to_keypoints = distance
            
            # Also consider the player's position relative to the court bounds
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            
            # More aggressive scoring for tennis players
            # Prefer players that are:
            # 1. Very close to court keypoints (stricter)
            # 2. Have optimal size for tennis players
            # 3. Are positioned in the optimal court area
            
            # Size scoring - prefer medium-large players (actual tennis players)
            size_penalty = 0
            optimal_area_min = 15000
            optimal_area_max = 60000
            if bbox_area < optimal_area_min:
                size_penalty = (optimal_area_min - bbox_area) / 1000  # Penalty for too small
            elif bbox_area > optimal_area_max:
                size_penalty = (bbox_area - optimal_area_max) / 2000  # Penalty for too large
                
            # Position penalty - prefer players in the court area
            position_penalty = 0
            frame_height = 1080
            frame_width = 1920
            
            # Prefer players in the central-lower area of the frame (tennis court area)
            if player_center[1] < frame_height * 0.4:  # Top 40% of frame (likely spectators)
                position_penalty = 3000
            elif player_center[1] < frame_height * 0.5:  # Upper-middle area
                position_penalty = 1000
                
            # Prefer players not too close to edges (likely spectators on sidelines)
            if player_center[0] < frame_width * 0.1 or player_center[0] > frame_width * 0.9:
                position_penalty += 1500
                
            # Court proximity penalty - must be reasonably close to court
            court_penalty = 0
            if min_distance_to_keypoints > 200:  # Stricter than before (was 300)
                court_penalty = min_distance_to_keypoints * 2
                
            total_score = min_distance_to_keypoints + size_penalty + position_penalty + court_penalty
            distances.append((track_id, total_score, min_distance_to_keypoints, bbox_area))
        
        # Sort by total score (lower is better)
        distances.sort(key=lambda x: x[1])
        
        # Only choose players if they meet strict criteria
        chosen_players = []
        for i, (track_id, total_score, court_distance, area) in enumerate(distances):
            if i >= 2:  # Only take top 2
                break
            # Stricter acceptance criteria
            if (court_distance < 200 and  # Must be very close to court
                total_score < 1000):      # Must have low total penalty score
                chosen_players.append(track_id)
        
        # If we found more than 2 good candidates, prefer the best 2
        # If we found fewer than 2 good candidates, return what we have
        return chosen_players[:2]


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            # Check if tracking ID is available
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
            else:
                # Use a fallback ID if tracking fails
                track_id = hash(tuple(box.xyxy.tolist()[0])) % 10000  # Generate pseudo-ID from bbox coordinates
                
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            confidence = box.conf.tolist()[0]
            
            # Only keep person detections with strict criteria for tennis courts
            if object_cls_name == "person":
                x1, y1, x2, y2 = result
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height
                
                # More aggressive filtering for tennis players:
                # 1. Higher confidence threshold
                # 2. Better size constraints (tennis players should be prominent)
                # 3. Better aspect ratio (standing people)
                # 4. Position filtering (lower part of frame = on court)
                min_confidence = 0.7  # Increased from 0.5
                min_area = 8000  # Increased from 2000 (larger players only)
                max_area = 120000  # Decreased from 200000 (filter very large detections)
                min_height = 80  # Increased from 50
                max_width = 200  # Add max width constraint
                aspect_ratio = bbox_height / bbox_width if bbox_width > 0 else 0
                
                # Position filtering - tennis players should be in lower 70% of frame
                frame_height = 1080  # Assuming HD video
                center_y = (y1 + y2) / 2
                
                if (confidence >= min_confidence and 
                    min_area <= bbox_area <= max_area and 
                    bbox_height >= min_height and
                    bbox_width <= max_width and
                    aspect_ratio >= 1.2 and  # People should be taller (increased from 0.8)
                    aspect_ratio <= 4.0 and  # But not too tall (avoid weird detections)
                    center_y >= frame_height * 0.3):  # Only lower 70% of frame
                    player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    