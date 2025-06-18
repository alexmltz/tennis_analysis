import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def read_video_limited(video_path, max_frames=None, start_frame=0):
    """
    Read video with memory efficiency options
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to read (None for all)
        start_frame: Frame number to start from
    
    Returns:
        List of frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames.append(frame)
        frame_count += 1
        
        # Stop if we've reached max frames
        if max_frames is not None and frame_count >= max_frames:
            break
            
    cap.release()
    return frames

def read_video_sampled(video_path, frame_step=10, max_frames=None, start_frame=0, end_frame=None):
    """
    Read video frames with sampling
    
    Args:
        video_path: Path to video file
        frame_step: Step size for sampling (e.g., 10 = every 10th frame)
        max_frames: Maximum number of frames to read (None for all)
        start_frame: Starting frame number (default: 0)
        end_frame: Ending frame number (None for end of video)
    
    Returns:
        List of sampled frames
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count if end_frame is not specified
    if end_frame is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = total_frames
    
    frames = []
    current_frame = start_frame
    
    while current_frame < end_frame:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        frames.append(frame)
        
        # Check if we've reached max_frames limit
        if max_frames is not None and len(frames) >= max_frames:
            break
            
        # Move to next sampled frame
        current_frame += frame_step
    
    cap.release()
    return frames

def get_video_info(video_path):
    """
    Get basic information about a video file
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration
    }

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()