# scripts/process_live_stream.py
import argparse
import os
import sys
import torch
import numpy as np
import cv2
import time
import math
from collections import defaultdict

# --- Add project root to sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# ------------------------------------

# --- Import project modules ---
# Use try-except for standalone mode compatibility
try:
    from utils.config_loader import load_app_config
    from utils.video_io import create_video_writer
    from utils.visualization import draw_tracked_results, draw_fps
    from utils.image_processing import crop_image_numpy
    from utils.gallery import ReIDGallery
    from models.detection_tracking import DetectorTracker
    from models.reid import ReIDModel
except ImportError as e:
    # Handle import error only if running as main script
    if __name__ == '__main__':
        print(f"Error: Could not import project modules in standalone mode. {e}")
        print("Ensure this script is run from the project root or PYTHONPATH is set correctly.")
        sys.exit(1)
    else: # Allow import to fail if imported by run.py (it handles imports)
        # Define dummy classes/functions or re-raise if needed for type hinting
        pass
# ---------------------------

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! HÀM CHÍNH ĐỂ XỬ LÝ LIVE STREAM (GỌI BỞI run.py) !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def run_live_stream_processing(config, detector_tracker, reid_model, gallery, args):
    """
    Runs the pipeline on a live video stream using pre-initialized models/gallery.

    Args:
        config (dict): The loaded application configuration.
        detector_tracker (DetectorTracker): Initialized detector/tracker model.
        reid_model (ReIDModel): Initialized ReID model.
        gallery (ReIDGallery): Initialized ReID gallery.
        args (argparse.Namespace): Parsed arguments specific to the live task.
    """
    reid_cfg = config['reid']
    # yolo_cfg = config['yolo'] # Có thể cần nếu có tham số yolo dùng ở đây

    # --- Lấy các tham số hiệu quả (đã được override bởi args trong run.py nếu có) ---
    target_fps = reid_cfg.get('target_fps')
    reid_batch_size = reid_cfg.get('reid_batch_size', 64)
    reid_interval = reid_cfg.get('reid_interval', 10)
    target_delay = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0

    print(f"Effective Settings: Target FPS={target_fps if target_delay > 0 else 'Max'}, ReID Batch={reid_batch_size}, ReID Interval={reid_interval}")
    print(f"Effective Re-ranking for Live Task: {reid_cfg['use_re_ranking']}") # In giá trị re-ranking hiệu quả

    # --- 3. Connect to Live Stream ---
    stream_url = args.stream_url # Lấy URL stream từ args (đã được xử lý bởi run.py nếu là link YT)
    print(f"Attempting to connect to live stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Could not open video stream. Check URL/network.")
        cap.release()
        return

    print("Successfully connected to stream.")

    # --- 4. Setup Output Video Writer (Optional) ---
    video_writer = None
    frame_width, frame_height = 0, 0
    output_fps = 30.0 # Luôn dùng FPS cố định cho output live stream

    # --- 5. Processing Loop ---
    frame_count = 0
    start_time_pipeline = time.time()
    last_time_loop = start_time_pipeline
    reid_fps_display = 0.0
    track_id_to_reid_id = {}

    print(f"Starting live stream processing...")
    if args.output: print(f" - Output will be saved to: {args.output}")
    if not args.no_display: print(f" - Press 'q' in the display window to quit.")

    # ============================
    #      LIVE STREAM LOOP
    # ============================
    try:
        while True:
            current_time_loop = time.time()
            loop_delta_time = current_time_loop - last_time_loop
            last_time_loop = current_time_loop
            overall_fps_display = 1.0 / loop_delta_time if loop_delta_time > 0 else 0

            # --- Read Frame ---
            ret, frame = cap.read()

            # --- Handle Stream Errors/End ---
            if not ret:
                print(f"Warning: Failed to grab frame {frame_count + 1}. Stream ended or issue occurred.")
                time.sleep(0.5) # Wait slightly
                ret, frame = cap.read() # Try one more time
                if not ret:
                    print("Failed again. Exiting loop.")
                    break # Exit if still failing

            frame_count += 1
            if frame is None: continue # Skip if frame is None after retry

            # --- Initialize Video Writer ---
            if video_writer is None and args.output:
                 if frame_height == 0 or frame_width == 0:
                      frame_height, frame_width = frame.shape[:2]
                      print(f"Stream Info detected: {frame_width}x{frame_height}")
                 if frame_width > 0 and frame_height > 0:
                      try:
                           video_writer = create_video_writer(args.output, frame_width, frame_height, output_fps)
                      except Exception as e:
                           print(f"Warning: Could not create video writer. Output disabled. Error: {e}")
                           args.output = None

            # --- Process Frame ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracking_results = detector_tracker.track_frame(frame_rgb)

            reid_processed_this_frame = False
            if tracking_results and tracking_results.boxes is not None and tracking_results.boxes.id is not None:
                try:
                    boxes_xyxy = tracking_results.boxes.xyxy.cpu().numpy()
                    track_ids_in_frame = tracking_results.boxes.id.cpu().numpy().astype(int)
                except Exception as e: track_ids_in_frame = []

                if len(track_ids_in_frame) > 0 and frame_count % reid_interval == 0:
                    reid_processed_this_frame = True
                    start_reid_time = time.time()
                    all_crops_for_reid = []
                    all_track_ids_for_reid = []

                    for i, track_id in enumerate(track_ids_in_frame):
                        bbox = boxes_xyxy[i]
                        crop = crop_image_numpy(frame_rgb, bbox)
                        if crop is not None:
                            all_crops_for_reid.append(crop)
                            all_track_ids_for_reid.append(track_id)

                    if all_crops_for_reid:
                        num_batches = math.ceil(len(all_crops_for_reid) / reid_batch_size)
                        for i_batch in range(num_batches):
                            start_idx = i_batch * reid_batch_size
                            end_idx = min((i_batch + 1) * reid_batch_size, len(all_crops_for_reid))
                            batch_crops = all_crops_for_reid[start_idx:end_idx]
                            batch_track_ids = all_track_ids_for_reid[start_idx:end_idx]
                            if not batch_crops: continue

                            query_features_batch = reid_model.extract_features_optimized(batch_crops)
                            if query_features_batch is not None:
                                assigned_reid_ids_batch = gallery.assign_ids(query_features_batch)
                                if len(assigned_reid_ids_batch) == len(batch_track_ids):
                                    for j, track_id in enumerate(batch_track_ids):
                                        if assigned_reid_ids_batch[j] != -1:
                                             track_id_to_reid_id[track_id] = assigned_reid_ids_batch[j]
                                else: print(f"CRITICAL WARNING: ReID results/tracks mismatch batch {i_batch+1}/{num_batches} frame {frame_count}.")

                    end_reid_time = time.time()
                    reid_processing_time = end_reid_time - start_reid_time
                    reid_fps_display = 1.0 / reid_processing_time if reid_processing_time > 1e-6 else 0

            # --- Visualization ---
            output_frame = frame.copy()
            output_frame = draw_tracked_results(output_frame, tracking_results, track_id_to_reid_id)
            output_frame = draw_fps(output_frame, overall_fps_display, reid_fps_display if reid_processed_this_frame else None)

            # --- Display Frame ---
            if not args.no_display:
                try:
                     cv2.imshow("Live Stream ReID - Press 'q' to Quit", output_frame)
                except Exception as e:
                     print(f"Error displaying frame: {e}. Disabling display.")
                     args.no_display = True

            # --- Write Frame ---
            if video_writer is not None:
                 try: video_writer.write(output_frame)
                 except Exception as e: print(f"Error writing frame {frame_count}: {e}")

            # --- Target FPS Delay ---
            current_frame_end_time = time.time()
            actual_processing_time = current_frame_end_time - current_time_loop
            wait_time = target_delay - actual_processing_time
            if wait_time > 0:
                time.sleep(wait_time)

            # --- Quit Key Check ---
            if not args.no_display:
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                      print("Processing stopped by user ('q' pressed).")
                      break

    except KeyboardInterrupt: print("\nProcessing interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn unexpected error occurred during live stream processing: {e}")
        import traceback; traceback.print_exc()
    finally:
        # --- Cleanup ---
        # ... (code cleanup giống như trong file main_pipeline.py) ...
        end_time_pipeline = time.time(); total_time = end_time_pipeline - start_time_pipeline
        avg_fps = frame_count / total_time if total_time > 1e-6 else 0
        print("\n--- Live Stream Processing Summary ---"); print(f"Processed approximately {frame_count} frames.")
        if total_time > 0: print(f"Total Processing Time: {total_time:.2f} seconds.")
        if avg_fps > 0: print(f"Average Overall FPS (incl. delay): {avg_fps:.2f}")
        if gallery: print(f"Final Gallery Size: {gallery.get_gallery_size()}")
        print("Releasing resources...");
        if cap is not None: cap.release()
        if video_writer is not None: video_writer.release()
        cv2.destroyAllWindows(); print("Processing finished.")

# --- Block để chạy standalone ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReID Pipeline on Live Stream (Standalone)")
    parser.add_argument('--stream-url', '-s', type=str, required=True, help='Direct stream URL (e.g., HLS .m3u8, RTSP) OR YouTube Live page URL.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output video file path.')
    parser.add_argument('--reid-config', type=str, default='configs/reid_config.yaml', help='ReID config path.')
    parser.add_argument('--yolo-config', type=str, default='configs/yolo_config.yaml', help='YOLO config path.')
    parser.add_argument('--no-display', action='store_true', help='Disable display window.')
    parser.add_argument('--target-fps', type=float, default=None, help='Limit processing FPS.')
    parser.add_argument('--reid-batch-size', type=int, default=None, help='ReID batch size.')
    parser.add_argument('--use-re-ranking', action=argparse.BooleanOptionalAction, default=False, help='Enable/disable re-ranking (Default: False).')
    args = parser.parse_args()

    # --- Tự động lấy URL stream nếu là link YouTube (cần streamlink) ---
    stream_url_to_use = args.stream_url
    if args.stream_url and ('https://www.youtube.com/watch?v=...5' in args.stream_url or 'https://www.youtube.com/watch?v=...6' in args.stream_url):
         print("[Standalone Mode] YouTube URL detected. Attempting to get direct stream URL via streamlink...")
         import subprocess
         try:
              process = subprocess.run(["streamlink", "--stream-url", args.stream_url, "best"], check=True, capture_output=True, text=True, encoding='utf-8')
              direct_url = process.stdout.strip()
              if direct_url and ('http' in direct_url or 'rtmp' in direct_url):
                   print(f"Found direct stream URL: {direct_url}")
                   stream_url_to_use = direct_url
              else:
                   print("Error: streamlink did not return a valid URL.")
                   sys.exit(1)
         except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
              print(f"Error using streamlink: {e}. Please ensure streamlink is installed and the YouTube URL is valid/live.")
              sys.exit(1)

    # --- Gán lại URL đã xử lý ---
    args.stream_url = stream_url_to_use # Hàm run_live_stream_processing sẽ dùng args.stream_url này

    # --- Load config và model trong standalone mode ---
    print("[Standalone Mode] Loading config and models for live stream...")
    try:
        config = load_app_config(args.reid_config, args.yolo_config)
        # Apply overrides from args
        if args.target_fps is not None: config['reid']['target_fps'] = args.target_fps
        if args.reid_batch_size is not None: config['reid']['reid_batch_size'] = args.reid_batch_size
        if args.use_re_ranking is not None: config['reid']['use_re_ranking'] = args.use_re_ranking

        detector_tracker = DetectorTracker(config['yolo'])
        reid_model = ReIDModel(config['reid'])
        gallery = ReIDGallery(config['reid'])
        print("[Standalone Mode] Models loaded.")
        run_live_stream_processing(config, detector_tracker, reid_model, gallery, args)
    except Exception as e:
         print(f"[Standalone Mode] Error: {e}")
         import traceback
         traceback.print_exc()