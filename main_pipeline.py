# main_pipeline.py
import cv2
import torch
import numpy as np
import time
import argparse
import os
from collections import defaultdict

# --- Import project modules ---
# Ensure the project root is discoverable if running script directly
# import sys
# if '.' not in sys.path: sys.path.append('.') # Add current dir for direct script run

from utils.config_loader import load_app_config
from utils.video_io import read_video_frames, create_video_writer
from utils.visualization import draw_tracked_results, draw_fps
from utils.image_processing import crop_image_numpy
from utils.gallery import ReIDGallery
from models.detection_tracking import DetectorTracker
from models.reid import ReIDModel
import torch._dynamo
torch._dynamo.config.suppress_errors = True
def run_video_pipeline(args):
    """Runs the full detection, tracking, and ReID pipeline on a video."""

    # --- 1. Load Configuration ---
    try:
        config = load_app_config(args.reid_config, args.yolo_config)
        reid_cfg = config['reid']
        yolo_cfg = config['yolo']
    except FileNotFoundError as e:
         print(f"Error: {e}")
         print("Please ensure config files exist and paths are correct.")
         return
    except Exception as e:
         print(f"Error loading configurations: {e}")
         return

    # --- 2. Initialize Models and Gallery ---
    try:
        print("Initializing models...")
        # Detector and Tracker (YOLO)
        detector_tracker = DetectorTracker(yolo_cfg)
        # Re-Identification Model (CLIP-ReID)
        reid_model = ReIDModel(reid_cfg)
        # ReID Gallery (using optimized tensor backend)
        gallery = ReIDGallery(reid_cfg)
        print("Models and Gallery initialized successfully.")
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, Exception) as e: # Catch potential init errors
         print(f"Error initializing models or gallery: {e}")
         print("Please check model paths, checkpoints, dependencies, and configurations.")
         return


    # --- 3. Setup Video I/O ---
    video_path = args.input
    output_path = args.output

    try:
         frame_gen = read_video_frames(video_path)
    except (FileNotFoundError, IOError) as e:
         print(f"Error opening video input: {e}")
         return

    video_writer = None
    frame_width, frame_height = 0, 0

    # Process first frame to get dimensions for writer and check video validity
    try:
        first_frame = next(frame_gen)
        if first_frame is None: raise StopIteration # Handle case where generator yields None
        frame_height, frame_width = first_frame.shape[:2]
        print(f"Video Info: {frame_width}x{frame_height}")
        # Recreate generator to start from the beginning again
        frame_gen = read_video_frames(video_path)
    except StopIteration:
        print(f"Error: Input video file {video_path} appears to be empty or corrupted.")
        return
    except Exception as e:
        print(f"Error reading first frame: {e}")
        return

    # Create Video Writer if output path is provided
    if output_path:
        try:
            # Attempt to get original FPS
            cap_temp = cv2.VideoCapture(video_path)
            fps = cap_temp.get(cv2.CAP_PROP_FPS) if cap_temp.isOpened() else 30.0
            cap_temp.release()
            if fps <= 0: fps = 30.0 # Use default if FPS read fails
            video_writer = create_video_writer(output_path, frame_width, frame_height, fps)
        except (IOError, ValueError, Exception) as e:
             print(f"Warning: Could not create video writer for {output_path}. Error: {e}")
             print("Output video will not be saved.")
             output_path = None # Disable saving


    # --- 4. Processing Loop Variables ---
    frame_count = 0
    start_time_pipeline = time.time()
    last_time_loop = start_time_pipeline
    reid_fps_display = 0.0 # For displaying instantaneous ReID FPS
    track_id_to_reid_id = {} # Mapping: Current Tracker ID -> Assigned Persistent ReID
    reid_interval = reid_cfg.get('reid_interval', 10) # How often to run ReID

    print(f"Starting video processing...")
    print(f" - Input: {video_path}")
    print(f" - Output: {output_path if output_path else 'Disabled'}")
    print(f" - ReID Interval: {reid_interval} frames")
    print(f" - Press 'q' in the display window to quit.")
    # ==============================================================
    #                     MAIN PROCESSING LOOP
    # ==============================================================
    try:
        for frame in frame_gen:
            frame_count += 1
            current_time_loop = time.time()
            loop_delta_time = current_time_loop - last_time_loop
            last_time_loop = current_time_loop
            overall_fps_display = 1.0 / loop_delta_time if loop_delta_time > 0 else 0

            # Ensure frame is valid
            if frame is None:
                print(f"Warning: Received None frame at index {frame_count}. Skipping.")
                continue

            # --- 1. Detection & Tracking ---
            # Convert frame to RGB for models that expect it (YOLOv8 works well with RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracking_results = detector_tracker.track_frame(frame_rgb)

            # --- 2. Re-Identification (Periodically) ---
            reid_processed_this_frame = False
            current_track_ids = [] # List of track IDs present in THIS frame

            # Proceed only if tracking was successful and produced IDs
            if tracking_results and tracking_results.boxes is not None and tracking_results.boxes.id is not None:
                try:
                    # Extract boxes and track IDs for tracks present in this frame
                    boxes_xyxy = tracking_results.boxes.xyxy.cpu().numpy()
                    track_ids_in_frame = tracking_results.boxes.id.cpu().numpy().astype(int)
                    current_track_ids = list(track_ids_in_frame) # Store IDs seen in this frame
                except Exception as e:
                    print(f"Error accessing tracking results boxes/IDs in frame {frame_count}: {e}")
                    track_ids_in_frame = [] # Prevent further processing if error


                # --- Run ReID at the specified interval if tracks exist ---
                if track_ids_in_frame is not None and len(track_ids_in_frame) > 0 and frame_count % reid_interval == 0:
                    reid_processed_this_frame = True
                    start_reid_time = time.time()
                    # print(f"--- Frame {frame_count}: Running Re-Identification ---") # Verbose Log

                    np_crops_to_reid = []
                    track_ids_for_reid = [] # Track IDs corresponding to the successfully cropped images

                    # Crop images using NumPy slicing for each tracked box
                    for i, track_id in enumerate(track_ids_in_frame):
                        bbox = boxes_xyxy[i]
                        crop = crop_image_numpy(frame_rgb, bbox) # Use utility for safe cropping
                        if crop is not None:
                            np_crops_to_reid.append(crop)
                            track_ids_for_reid.append(track_id)
                        # else: print(f"Skipped invalid crop for track {track_id}")

                    if np_crops_to_reid:
                        # Extract features in batch using the optimized method
                        query_features_batch = reid_model.extract_features_optimized(np_crops_to_reid)

                        if query_features_batch is not None:
                            # Compare features with gallery and get assigned IDs
                            assigned_reid_ids = gallery.assign_ids(query_features_batch)

                            # Update the track_id -> reid_id mapping for the processed tracks
                            if len(assigned_reid_ids) == len(track_ids_for_reid):
                                for i, track_id in enumerate(track_ids_for_reid):
                                    if assigned_reid_ids[i] != -1: # Check if assignment was successful
                                         track_id_to_reid_id[track_id] = assigned_reid_ids[i]
                                    # else: print(f"Warning: Gallery assignment returned -1 for track {track_id}")
                            else:
                                 # This indicates a potential issue in feature extraction or gallery logic
                                 print(f"CRITICAL WARNING: Mismatch between processed ReID results ({len(assigned_reid_ids)}) and tracks sent for ReID ({len(track_ids_for_reid)}) in frame {frame_count}. Check logs.")
                        # else: print(f"Feature extraction returned None for frame {frame_count}.")


                    # --- Calculate ReID processing time and FPS for display ---
                    end_reid_time = time.time()
                    reid_processing_time = end_reid_time - start_reid_time
                    # Avoid division by zero; display 0 FPS if time is negligible or negative
                    reid_fps_display = 1.0 / reid_processing_time if reid_processing_time > 1e-6 else 0
                    # print(f"--- ReID Time: {reid_processing_time:.4f}s | ReID FPS: {reid_fps_display:.2f} ---") # Verbose Log


            # --- 3. Visualization ---
            output_frame = frame.copy() # Draw on a copy to preserve original frame if needed
            # Draw tracking results (boxes and assigned ReID labels)
            output_frame = draw_tracked_results(output_frame, tracking_results, track_id_to_reid_id)
            # Draw FPS info
            output_frame = draw_fps(output_frame, overall_fps_display, reid_fps_display if reid_processed_this_frame else None)

            # --- 4. Display/Save ---
            if not args.no_display:
                try:
                     cv2.imshow("Person ReID Pipeline - Press 'q' to Quit", output_frame)
                     # Wait for 1ms, check if 'q' is pressed
                     if cv2.waitKey(1) & 0xFF == ord('q'):
                         print("Processing stopped by user ('q' pressed).")
                         break
                except Exception as e:
                     print(f"Error displaying frame: {e}. Check if GUI is available.")
                     # Consider adding a flag to disable display automatically if it fails once
                     args.no_display = True # Disable display for future frames

            if video_writer is not None:
                 try:
                     video_writer.write(output_frame)
                 except Exception as e:
                     print(f"Error writing frame {frame_count} to video file: {e}")
                     # Consider stopping or disabling writer after multiple errors
                     # video_writer.release()
                     # video_writer = None

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ==============================================================
        #                      CLEANUP ACTIONS
        # ==============================================================
        end_time_pipeline = time.time()
        total_time = end_time_pipeline - start_time_pipeline
        # Avoid division by zero if total_time is very small or negative
        avg_fps = frame_count / total_time if total_time > 1e-6 else 0

        print("\n--- Pipeline Summary ---")
        print(f"Processed {frame_count} frames.")
        if total_time > 0: print(f"Total Processing Time: {total_time:.2f} seconds.")
        if avg_fps > 0: print(f"Average Overall FPS: {avg_fps:.2f}")
        print(f"Final Gallery Size: {gallery.get_gallery_size()} unique IDs.")

        # Release resources
        if video_writer is not None:
            print("Releasing video writer...")
            video_writer.release()
            if output_path: print(f"Output video saved to: {output_path}")
        cv2.destroyAllWindows()
        print("Processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Person Detection, Tracking, and Re-Identification Pipeline on Video")
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', '-o', type=str, default='output/result_video.mp4', help='Path to save the output video file.')
    parser.add_argument('--reid-config', type=str, default='configs/reid_config.yaml', help='Path to the ReID configuration file.')
    parser.add_argument('--yolo-config', type=str, default='configs/yolo_config.yaml', help='Path to the YOLO/tracking configuration file.')
    parser.add_argument('--no-display', action='store_true', help='Do not display the video window during processing.')

    args = parser.parse_args()

    # Basic validation of input path
    if not os.path.isfile(args.input):
        print(f"Error: Input video file not found at '{args.input}'")
    else:
        run_video_pipeline(args)