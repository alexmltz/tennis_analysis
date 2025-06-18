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
        # Instead of only using the first frame, analyze multiple frames to find the best players
        # This handles cases where actual tennis players appear/disappear while spectators remain static
        
        # Collect player statistics across all frames
        player_stats = {}
        
        for frame_idx, player_dict in enumerate(player_detections[:10]):  # Analyze first 10 frames
            frame_candidates = self.choose_players(court_keypoints, player_dict)
            
            for player_id in frame_candidates:
                if player_id not in player_stats:
                    player_stats[player_id] = {
                        'appearances': 0,
                        'total_score': 0,
                        'positions': [],
                        'first_appearance': frame_idx
                    }
                
                player_stats[player_id]['appearances'] += 1
                
                # Calculate score for this player in this frame
                if player_id in player_dict:
                    bbox = player_dict[player_id]
                    x1, y1, x2, y2 = bbox
                    player_center = get_center_of_bbox(bbox)
                    
                    # Court-based scoring
                    frame_width, frame_height = 1920, 1080
                    center_x, center_y = player_center
                    
                    score = 0
                    # Prefer players in court area
                    if frame_width * 0.2 < center_x < frame_width * 0.8:
                        score += 3
                    if center_y > frame_height * 0.2:
                        score += 2
                    if (x2-x1) * (y2-y1) > 15000:  # Substantial size
                        score += 2
                    
                    player_stats[player_id]['total_score'] += score
                    player_stats[player_id]['positions'].append(player_center)
        
        # Calculate movement variance (tennis players move, spectators are static)
        for player_id, stats in player_stats.items():
            if len(stats['positions']) > 1:
                positions = stats['positions']
                # Calculate variance in x and y positions
                x_positions = [pos[0] for pos in positions]
                y_positions = [pos[1] for pos in positions]
                
                x_variance = sum((x - sum(x_positions)/len(x_positions))**2 for x in x_positions) / len(x_positions)
                y_variance = sum((y - sum(y_positions)/len(y_positions))**2 for y in y_positions) / len(y_positions)
                
                # Bonus for movement (tennis players should move)
                movement_bonus = min(5, (x_variance + y_variance) / 1000)
                stats['total_score'] += movement_bonus
        
        # Select the best 2 players based on combined criteria
        player_rankings = []
        for player_id, stats in player_stats.items():
            avg_score = stats['total_score'] / max(1, stats['appearances'])
            # Prefer players who appear frequently and have good scores
            final_score = avg_score * (stats['appearances'] / 10)  # Normalize by max possible appearances
            
            player_rankings.append((player_id, final_score, stats['appearances']))
        
        # Sort by final score (descending)
        player_rankings.sort(key=lambda x: -x[1])
        
        # Take top 2 players
        chosen_players = [player_rankings[i][0] for i in range(min(2, len(player_rankings)))]
        
        print(f"Player selection analysis:")
        for player_id, score, appearances in player_rankings[:5]:  # Show top 5
            print(f"  Player {player_id}: score={score:.2f}, appearances={appearances}/10")
        print(f"  Chosen players: {chosen_players}")
        
        # Filter all frames based on chosen players
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        if len(player_dict) == 0:
            return []
        if len(player_dict) == 1:
            return list(player_dict.keys())
        if len(player_dict) == 2:
            return list(player_dict.keys())
        
        # For more than 2 players, we need intelligent selection
        # Focus on players who are:
        # 1. In the court area (not spectators)
        # 2. Have reasonable size (actual players, not distant people)
        # 3. Are positioned like tennis players (center court area)
        
        candidates = []
        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            player_center = get_center_of_bbox(bbox)
            player_foot = get_foot_position(bbox)
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            
            # Tennis court area analysis (assuming 1920x1080 frame)
            frame_width, frame_height = 1920, 1080
            
            # Tennis players should be:
            # - In central 60% of frame width (not on extreme edges)
            # - In lower 80% of frame height (on court, not in stands)
            # - Have substantial size (not tiny distant spectators)
            # - Not be static edge detections
            
            center_x, center_y = player_center
            
            # Court area constraints
            is_in_court_width = (frame_width * 0.2 < center_x < frame_width * 0.8)  # Central 60%
            is_in_court_height = (center_y > frame_height * 0.2)  # Lower 80%
            is_substantial_size = (bbox_area > 15000)  # Larger than small spectators
            is_not_edge = (x1 > 50 and x2 < frame_width - 50)  # Not touching frame edges
            is_reasonable_aspect = (1.0 < bbox_height/bbox_width < 4.0)  # Human proportions
            
            # Calculate distance to court center (approximate tennis court center)
            court_center_x, court_center_y = frame_width * 0.5, frame_height * 0.6
            distance_to_court_center = measure_distance(player_center, (court_center_x, court_center_y))
            
            # Score each candidate
            score = 0
            if is_in_court_width: score += 3
            if is_in_court_height: score += 2  
            if is_substantial_size: score += 2
            if is_not_edge: score += 1
            if is_reasonable_aspect: score += 1
            
            # Bonus for being closer to court center
            score += max(0, 3 - distance_to_court_center / 200)
            
            candidates.append((track_id, bbox, score, distance_to_court_center))
        
        # Sort by score (descending) then by distance to court center (ascending)
        candidates.sort(key=lambda x: (-x[2], x[3]))
        
        # Take top 2 candidates
        chosen_players = [candidates[i][0] for i in range(min(2, len(candidates)))]
        
        return chosen_players

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
        # Use more persistent tracking parameters
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
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
            
            # Balanced filtering for tennis players - less aggressive to catch more players
            if object_cls_name == "person":
                x1, y1, x2, y2 = result
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height
                
                # More balanced filtering to catch both players:
                # 1. Lower confidence threshold to catch partially occluded players
                # 2. Wider size range to accommodate different distances
                # 3. Better aspect ratio range
                # 4. Less strict position filtering
                min_confidence = 0.5  # Reduced from 0.7 to catch more players
                min_area = 5000  # Reduced from 8000 to catch smaller/distant players
                max_area = 150000  # Increased from 120000 for closer players
                min_height = 60  # Reduced from 80
                max_width = 250  # Increased from 200
                aspect_ratio = bbox_height / bbox_width if bbox_width > 0 else 0
                
                # Less strict position filtering - allow players in more areas
                frame_height = 1080
                center_y = (y1 + y2) / 2
                
                if (confidence >= min_confidence and 
                    min_area <= bbox_area <= max_area and 
                    bbox_height >= min_height and
                    bbox_width <= max_width and
                    aspect_ratio >= 1.0 and  # Reduced from 1.2 to catch more orientations
                    aspect_ratio <= 5.0 and  # Increased from 4.0
                    center_y >= frame_height * 0.2):  # Allow top 20% (was 30%) for players near net
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


    