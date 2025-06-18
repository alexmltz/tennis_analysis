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

def get_video_info(video_path):
    """Get basic video information"""
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