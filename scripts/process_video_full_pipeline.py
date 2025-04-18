# main_pipeline.py
import cv2
import torch
import numpy as np
import time
import argparse
import os
from collections import defaultdict
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# --- Import project modules ---
# Đảm bảo có thể chạy độc lập
try:
    from utils.config_loader import load_app_config
    from utils.video_io import read_video_frames, create_video_writer
    from utils.visualization import draw_tracked_results, draw_fps
    from utils.image_processing import crop_image_numpy
    from utils.gallery import ReIDGallery
    from models.detection_tracking import DetectorTracker
    from models.reid import ReIDModel
except ImportError:
     # Xử lý trường hợp chạy từ thư mục khác hoặc cấu trúc thay đổi
     if __name__ == '__main__': # Chỉ báo lỗi nếu đang chạy file này trực tiếp
         print("Error: Could not import project modules in standalone mode.")
         print("Ensure this script is run from the project root or add project root to PYTHONPATH.")
         sys.exit(1)
     else: # Nếu được import bởi file khác (như run.py), bỏ qua lỗi ở đây
         pass


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! HÀM CHÍNH ĐỂ XỬ LÝ VIDEO (ĐƯỢC GỌI BỞI run.py) !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def run_video_processing(config, detector_tracker, reid_model, gallery, args):
    """
    Runs the video processing loop using pre-initialized models and gallery.

    Args:
        config (dict): The loaded application configuration.
        detector_tracker (DetectorTracker): Initialized detector/tracker model.
        reid_model (ReIDModel): Initialized ReID model.
        gallery (ReIDGallery): Initialized ReID gallery.
        args (argparse.Namespace): Parsed arguments specific to the video task.
    """
    reid_cfg = config['reid']
    # yolo_cfg = config['yolo'] # Không cần dùng trực tiếp ở đây nữa

    # --- Setup Video I/O ---
    video_path = args.input
    output_path = args.output

    try:
        frame_gen = read_video_frames(video_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error opening video input: {e}")
        return # Thoát nếu không mở được video

    video_writer = None
    frame_width, frame_height = 0, 0

    # Process first frame to get dimensions for writer
    try:
        first_frame = next(frame_gen)
        if first_frame is None: raise StopIteration
        frame_height, frame_width = first_frame.shape[:2]
        print(f"Video Info: {frame_width}x{frame_height}")
        # Recreate generator to start from the beginning
        frame_gen = read_video_frames(video_path)
    except StopIteration:
        print(f"Error: Input video file {video_path} appears to be empty.")
        return
    except Exception as e:
        print(f"Error reading first frame: {e}")
        return

    # Create Video Writer
    if output_path:
        try:
            cap_temp = cv2.VideoCapture(video_path)
            fps = cap_temp.get(cv2.CAP_PROP_FPS) if cap_temp.isOpened() else 30.0
            cap_temp.release()
            if fps <= 0: fps = 30.0
            video_writer = create_video_writer(output_path, frame_width, frame_height, fps)
        except (IOError, ValueError, Exception) as e:
             print(f"Warning: Could not create video writer for {output_path}. Error: {e}")
             output_path = None

    # --- Processing Loop Variables ---
    frame_count = 0
    start_time_pipeline = time.time()
    last_time_loop = start_time_pipeline
    reid_fps_display = 0.0
    track_id_to_reid_id = {}
    reid_interval = reid_cfg.get('reid_interval', 10)

    print(f"Starting video processing loop (using pre-loaded models)...")
    print(f" - Input: {video_path}")
    print(f" - Output: {output_path if output_path else 'Disabled'}")
    print(f" - ReID Interval: {reid_interval} frames")
    if not args.no_display: print(f" - Press 'q' in the display window to quit.")
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

            if frame is None:
                print(f"Warning: Frame {frame_count} is None. Skipping.")
                continue

            # --- 1. Detection & Tracking ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracking_results = detector_tracker.track_frame(frame_rgb) # Sử dụng model đã truyền vào

            # --- 2. Re-Identification (Periodically) ---
            reid_processed_this_frame = False
            current_track_ids_in_frame = []

            if tracking_results and tracking_results.boxes is not None and tracking_results.boxes.id is not None:
                try:
                    boxes_xyxy = tracking_results.boxes.xyxy.cpu().numpy()
                    track_ids_in_frame = tracking_results.boxes.id.cpu().numpy().astype(int)
                except Exception as e:
                    print(f"Error accessing tracking results in frame {frame_count}: {e}")
                    track_ids_in_frame = []

                if len(track_ids_in_frame) > 0 and frame_count % reid_interval == 0:
                    reid_processed_this_frame = True
                    start_reid_time = time.time()

                    np_crops_to_reid = []
                    track_ids_for_reid = []

                    for i, track_id in enumerate(track_ids_in_frame):
                        bbox = boxes_xyxy[i]
                        crop = crop_image_numpy(frame_rgb, bbox)
                        if crop is not None:
                            np_crops_to_reid.append(crop)
                            track_ids_for_reid.append(track_id)

                    if np_crops_to_reid:
                        # Sử dụng reid_model đã truyền vào
                        query_features_batch = reid_model.extract_features_optimized(np_crops_to_reid)

                        if query_features_batch is not None:
                            # Sử dụng gallery đã truyền vào
                            assigned_reid_ids = gallery.assign_ids(query_features_batch)

                            if len(assigned_reid_ids) == len(track_ids_for_reid):
                                for i, track_id in enumerate(track_ids_for_reid):
                                    if assigned_reid_ids[i] != -1:
                                         track_id_to_reid_id[track_id] = assigned_reid_ids[i]
                            else:
                                 print(f"CRITICAL WARNING: ReID results/tracks mismatch frame {frame_count}.")

                    end_reid_time = time.time()
                    reid_processing_time = end_reid_time - start_reid_time
                    reid_fps_display = 1.0 / reid_processing_time if reid_processing_time > 1e-6 else 0

            # --- 3. Visualization ---
            output_frame = frame.copy()
            output_frame = draw_tracked_results(output_frame, tracking_results, track_id_to_reid_id)
            output_frame = draw_fps(output_frame, overall_fps_display, reid_fps_display if reid_processed_this_frame else None)

            # --- 4. Display/Save ---
            if not args.no_display:
                try:
                     cv2.imshow("Person ReID Pipeline - Press 'q' to Quit", output_frame)
                     if cv2.waitKey(1) & 0xFF == ord('q'):
                         print("Processing stopped by user ('q' pressed).")
                         break
                except Exception as e:
                     print(f"Error displaying frame: {e}. Disabling display.")
                     args.no_display = True

            if video_writer is not None:
                 try:
                     video_writer.write(output_frame)
                 except Exception as e:
                     print(f"Error writing frame {frame_count}: {e}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 5. Cleanup ---
        end_time_pipeline = time.time()
        total_time = end_time_pipeline - start_time_pipeline
        avg_fps = frame_count / total_time if total_time > 1e-6 else 0
        print("\n--- Video Processing Summary ---")
        print(f"Processed {frame_count} frames.")
        if total_time > 0: print(f"Total Processing Time: {total_time:.2f} seconds.")
        if avg_fps > 0: print(f"Average Overall FPS: {avg_fps:.2f}")
        if gallery: print(f"Final Gallery Size: {gallery.get_gallery_size()} unique IDs.") # Check if gallery exists
        if video_writer is not None:
            print("Releasing video writer...")
            video_writer.release()
            if output_path: print(f"Output video saved to: {output_path}")
        cv2.destroyAllWindows()
        print("Video processing finished.")

# ==============================================================
#     BLOCK ĐỂ CHẠY FILE NÀY ĐỘC LẬP (Standalone Mode)
# ==============================================================
if __name__ == "__main__":
    # --- Argument Parser cho chế độ Standalone ---
    parser = argparse.ArgumentParser(description="Run Person ReID Pipeline on Video (Standalone Script)")
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', '-o', type=str, default='output/main_pipeline_output.mp4', help='Path to save the output video file.')
    parser.add_argument('--reid-config', type=str, default='configs/reid_config.yaml', help='Path to the ReID configuration file.')
    parser.add_argument('--yolo-config', type=str, default='configs/yolo_config.yaml', help='Path to the YOLO/tracking configuration file.')
    parser.add_argument('--no-display', action='store_true', help='Do not display the video window.')
    args = parser.parse_args()

    # --- Load Configs và Models (Load lại từ đầu) ---
    if not os.path.isfile(args.input):
        print(f"Error: Input video file not found at '{args.input}'")
    else:
        print("[Standalone Mode] Loading configs and models...")
        try:
            # Load config
            config = load_app_config(args.reid_config, args.yolo_config)
            # Initialize models
            detector_tracker = DetectorTracker(config['yolo'])
            reid_model = ReIDModel(config['reid'])
            gallery = ReIDGallery(config['reid'])
            print("[Standalone Mode] Models and Gallery loaded.")
            # Gọi hàm xử lý chính
            run_video_processing(config, detector_tracker, reid_model, gallery, args)
        except ImportError:
             print("Import Error in Standalone Mode. Make sure running from project root or PYTHONPATH is set.")
        except (FileNotFoundError, RuntimeError, ValueError, Exception) as e:
             print(f"[Standalone Mode] Error during initialization or processing: {e}")
             import traceback
             traceback.print_exc()