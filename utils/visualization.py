# utils/visualization.py
import cv2
import numpy as np
import random
from collections import defaultdict

# Use defaultdict to avoid KeyError if ID not seen before
id_color_map = defaultdict(lambda: (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
random.seed(42) # For consistent 'random' colors across runs

def get_color_for_id(person_id):
    """Generates or retrieves a consistent color for a given ID using defaultdict."""
    # The color is generated automatically by the defaultdict if the key is new
    return id_color_map[person_id]

def draw_tracked_results(frame, tracking_results, track_id_to_reid_id_map):
    """
    Draws bounding boxes and assigned ReID labels based on tracking results.

    Args:
        frame (numpy.ndarray): The frame to draw on (BGR format).
        tracking_results (ultralytics.engine.results.Results): Output from YOLO model.track().
        track_id_to_reid_id_map (dict): Mapping from tracker ID to assigned ReID.
    """
    output_frame = frame # Draw directly on the input frame reference

    # Check if tracking_results has valid boxes and IDs
    if tracking_results is None or tracking_results.boxes is None or tracking_results.boxes.id is None:
        # print("No tracking results or IDs to draw.") # Optional log
        return output_frame # Return original frame if no tracks

    try:
         boxes_xyxy = tracking_results.boxes.xyxy.cpu().numpy()
         track_ids = tracking_results.boxes.id.cpu().numpy().astype(int)
         # confidences = tracking_results.boxes.conf.cpu().numpy() # Optional: Get confidences
    except Exception as e:
         print(f"Error accessing tracking results properties: {e}")
         return output_frame # Return original frame on error

    for i, track_id in enumerate(track_ids):
        assigned_id = track_id_to_reid_id_map.get(track_id, None) # Get ReID if assigned

        # --- Only draw boxes for tracks that have been assigned a ReID ---
        # --- Modify this if you want to see unassigned tracks too ---
        if assigned_id is not None:
            box = boxes_xyxy[i]
            # confidence = confidences[i] # Optional
            x1, y1, x2, y2 = map(int, box)

            # Ensure coordinates are valid before drawing
            if x1 >= x2 or y1 >= y2: continue

            color = get_color_for_id(assigned_id)
            label = f"ID {assigned_id}" # Basic label
            # label = f"ID {assigned_id} C:{confidence:.2f}" # Optional label with confidence

            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # Calculate text size for background
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Position label slightly above the top-left corner, ensuring it's within frame
            label_x = x1
            label_y = max(y1 - 10, label_height + 5) # Position above, ensure space from top

            # Draw filled rectangle background for label
            cv2.rectangle(output_frame, (label_x, label_y - label_height - baseline),
                          (label_x + label_width, label_y + baseline//2), color, cv2.FILLED)

            # Draw label text (white for better contrast on colored background)
            cv2.putText(output_frame, label, (label_x, label_y - baseline//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output_frame


def draw_fps(frame, overall_fps, reid_fps=None):
    """Draws FPS information on the top-left corner of the frame."""
    # Ensure frame is writable if needed (usually is)
    # if not frame.flags['WRITEABLE']: frame = frame.copy()

    fps_text = f"FPS: {overall_fps:.2f}"
    cv2.putText(frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2, cv2.LINE_AA) # Red color

    if reid_fps is not None and reid_fps > 0:
         reid_fps_text = f"ReID FPS: {reid_fps:.2f}"
         cv2.putText(frame, reid_fps_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                     0.8, (0, 255, 0), 2, cv2.LINE_AA) # Green color
    return frame

def draw_single_bbox(image, bbox, label=None, color=(0, 255, 0), thickness=2):
    """Draws a single bounding box and label on an image (used for folder processing)."""
    output_image = image.copy() # Work on a copy
    x1, y1, x2, y2 = map(int, bbox[:4])

    # Draw bounding box
    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)

    # Draw label if provided
    if label:
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Position label above bbox
        label_y = max(y1 - 5, label_height + 5)
        # Draw background rectangle
        cv2.rectangle(output_image, (x1, label_y - label_height - baseline),
                      (x1 + label_width, label_y), color, cv2.FILLED)
        # Draw text
        cv2.putText(output_image, label, (x1, label_y - baseline // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA) # Black text
    return output_image