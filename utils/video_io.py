# utils/video_io.py
import cv2
import os

def read_video_frames(video_path):
    """Generator function to read frames from a video file."""
    if not os.path.exists(video_path):
         raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release() # Release resource if opened but failed
        raise IOError(f"Cannot open video file: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        yield frame # Trả về từng frame (BGR format)

    print(f"Finished reading {frame_count} frames from {video_path}")
    cap.release()

def create_video_writer(output_path, frame_width, frame_height, fps=30):
    """Creates a VideoWriter object to save video."""
    if not output_path:
         raise ValueError("Output path cannot be empty for video writer.")
    if frame_width <= 0 or frame_height <= 0:
         raise ValueError(f"Invalid frame dimensions for video writer: {frame_width}x{frame_height}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
         print(f"Creating output directory: {output_dir}")
         os.makedirs(output_dir)

    # Determine codec based on file extension
    if output_path.lower().endswith(".mp4"):
         # MP4V is widely compatible for .mp4
         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_path.lower().endswith(".avi"):
         # XVID or DIVX often used for .avi
         fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
         print(f"Warning: Unknown video extension for {output_path}. Using MP4V codec.")
         fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (int(frame_width), int(frame_height)))

    if not writer.isOpened():
         # Attempt fallback codec? (Less common)
         # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
         # writer = cv2.VideoWriter(output_path, fourcc, float(fps), (int(frame_width), int(frame_height)))
         # if not writer.isOpened():
         raise IOError(f"Cannot create video writer for path: {output_path} with codec {fourcc}")

    print(f"Video writer created for {output_path} ({frame_width}x{frame_height} @ {fps:.2f}fps)")
    return writer