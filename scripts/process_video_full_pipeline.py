# scripts/process_video_full_pipeline.py
import cv2
import torch
import numpy as np
import time
import argparse
import os
from collections import defaultdict
import torch._dynamo # <<< DÒNG BẠN ĐÃ THÊM >>>
torch._dynamo.config.suppress_errors = True # <<< DÒNG BẠN ĐÃ THÊM >>>
import math
import sys

# --- Import project modules ---
try:
    # Sử dụng import tuyệt đối (không có ..)
    from utils.config_loader import load_app_config
    from utils.video_io import read_video_frames, create_video_writer
    from utils.visualization import draw_tracked_results, draw_fps
    from utils.image_processing import crop_image_numpy
    from utils.gallery import ReIDGallery
    from models.detection_tracking import DetectorTracker
    from models.reid import ReIDModel
except ImportError:
     # Fallback cho standalone mode
     if __name__ == '__main__':
         project_root_for_standalone = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
         if project_root_for_standalone not in sys.path:
             sys.path.append(project_root_for_standalone)
         from utils.config_loader import load_app_config
         from utils.video_io import read_video_frames, create_video_writer
         from utils.visualization import draw_tracked_results, draw_fps
         from utils.image_processing import crop_image_numpy
         from utils.gallery import ReIDGallery
         from models.detection_tracking import DetectorTracker
         from models.reid import ReIDModel
     else: # Nếu được import bởi run.py, lỗi import ở đây là nghiêm trọng
          print("Error: Could not perform relative import in process_video_full_pipeline.py when imported.")
          raise

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! HÀM CHÍNH ĐỂ XỬ LÝ VIDEO (ĐƯỢC GỌI BỞI run.py) !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def run_video_processing(config, detector_tracker, reid_model, gallery, args):
    """
    Runs the video processing loop using pre-initialized models and gallery.
    Processes video at maximum possible speed.
    """
    reid_cfg = config['reid']
    yolo_cfg = config['yolo'] # Có thể dùng sau này

    # --- Lấy các config cần thiết ---
    reid_batch_size = reid_cfg.get('reid_batch_size', 64)
    reid_interval = reid_cfg.get('reid_interval', 10)
    print(f"Processing Parameters: ReID Interval={reid_interval}, ReID Batch Size={reid_batch_size}")
    # --- ĐÃ XÓA target_fps và target_delay ---

    # --- Setup Video I/O ---
    video_path = args.input
    output_path = args.output
    try:
        frame_gen = read_video_frames(video_path)
        first_frame = next(frame_gen, None) # Thêm None để xử lý video rỗng
        if first_frame is None:
             print(f"Error: Input video file {video_path} appears to be empty or cannot be read.")
             return
        frame_height, frame_width = first_frame.shape[:2]
        print(f"Video Info: {frame_width}x{frame_height}")
        frame_gen = read_video_frames(video_path) # Reset generator
        video_writer = None
        if output_path:
            cap_temp = cv2.VideoCapture(video_path)
            fps = cap_temp.get(cv2.CAP_PROP_FPS) if cap_temp.isOpened() else 30.0
            cap_temp.release()
            if fps <= 0: fps = 30.0
            video_writer = create_video_writer(output_path, frame_width, frame_height, fps)
    except (FileNotFoundError, IOError, StopIteration, Exception) as e:
        print(f"Error setting up video I/O: {e}")
        return

    # --- Xử lý "nháp" frame đầu tiên ---
    first_frame_processed = False
    if first_frame is not None:
        print("\n--- Processing first frame (initialization/warmup) ---")
        try:
            frame_count = 1 # Đặt frame_count là 1 cho frame đầu
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            _ = detector_tracker.track_frame(first_frame_rgb)
            # Chạy ReID thử (không cần thiết phải lấy đúng crop)
            with torch.no_grad():
                dummy_crops = [np.zeros((64, 32, 3), dtype=np.uint8)]
                _ = reid_model.extract_features_optimized(dummy_crops)
                if gallery.get_gallery_size() > 0:
                    dummy_features = torch.zeros((1, config['reid']['expected_feature_dim']), device=config['reid']['device'])
                    _ = gallery.assign_ids(dummy_features)
            first_frame_processed = True
            print("--- First frame processing finished ---")
        except Exception as e:
             print(f"Warning: Error during first frame processing: {e}. Continuing...")

    # --- Processing Loop Variables ---
    frame_count = 0 # Reset frame_count cho vòng lặp chính nếu frame đầu được xử lý
    if first_frame_processed: frame_count = 1 # Hoặc bắt đầu từ 1 nếu muốn giữ đúng số frame

    start_time_pipeline = time.time() # Bắt đầu tính giờ từ đây
    last_time_loop = start_time_pipeline
    reid_fps_display = 0.0
    track_id_to_reid_id = {}

    print(f"\nStarting main video processing loop...")
    if not args.no_display: print(f" - Press 'q' in the display window to quit.")
    # ==============================================================
    #                     MAIN PROCESSING LOOP
    # ==============================================================
    try:
        for frame in frame_gen: # Bắt đầu từ frame thứ 2 nếu frame đầu đã xử lý
            frame_count += 1
            current_time_loop = time.time()
            loop_processing_time = current_time_loop - last_time_loop
            last_time_loop = current_time_loop

            if frame is None: continue

            # --- 1. Detection & Tracking ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracking_results = detector_tracker.track_frame(frame_rgb)

            # --- 2. Re-Identification (Periodically) ---
            reid_processed_this_frame = False
            if tracking_results and tracking_results.boxes is not None and tracking_results.boxes.id is not None:
                try:
                    boxes_xyxy = tracking_results.boxes.xyxy.cpu().numpy()
                    track_ids_in_frame = tracking_results.boxes.id.cpu().numpy().astype(int)
                except Exception as e: track_ids_in_frame = []

                if len(track_ids_in_frame) > 0 and frame_count % reid_interval == 0:
                    reid_processed_this_frame = True
                    start_reid_time = time.time()
                    all_crops_for_reid, all_track_ids_for_reid = [], []
                    for i, track_id in enumerate(track_ids_in_frame):
                        bbox = boxes_xyxy[i]; crop = crop_image_numpy(frame_rgb, bbox)
                        if crop is not None:
                            all_crops_for_reid.append(crop); all_track_ids_for_reid.append(track_id)

                    if all_crops_for_reid:
                        num_batches = math.ceil(len(all_crops_for_reid) / reid_batch_size)
                        for i_batch in range(num_batches):
                            start_idx, end_idx = i_batch * reid_batch_size, min((i_batch + 1) * reid_batch_size, len(all_crops_for_reid))
                            batch_crops, batch_track_ids = all_crops_for_reid[start_idx:end_idx], all_track_ids_for_reid[start_idx:end_idx]
                            if not batch_crops: continue
                            query_features_batch = reid_model.extract_features_optimized(batch_crops)
                            if query_features_batch is not None:
                                assigned_reid_ids_batch = gallery.assign_ids(query_features_batch)
                                if len(assigned_reid_ids_batch) == len(batch_track_ids):
                                    for j, track_id in enumerate(batch_track_ids):
                                        if assigned_reid_ids_batch[j] != -1: track_id_to_reid_id[track_id] = assigned_reid_ids_batch[j]
                                else: print(f"CRITICAL WARNING: ReID results/tracks mismatch batch {i_batch+1}/{num_batches} frame {frame_count}.")
                    end_reid_time = time.time()
                    reid_processing_time = end_reid_time - start_reid_time
                    reid_fps_display = 1.0 / reid_processing_time if reid_processing_time > 1e-6 else 0

            # --- 3. Visualization ---
            output_frame = frame.copy()
            output_frame = draw_tracked_results(output_frame, tracking_results, track_id_to_reid_id)
            overall_fps_display = 1.0 / loop_processing_time if loop_processing_time > 0 else 0
            output_frame = draw_fps(output_frame, overall_fps_display, reid_fps_display if reid_processed_this_frame else None)

            # --- 4. Display/Save ---
            display_frame = output_frame
            if not args.no_display:
                try:
                    display_width = 1280 # Hoặc giá trị khác bạn muốn
                    if output_frame.shape[1] > display_width: # Chỉ resize nếu ảnh lớn
                        display_height = int(output_frame.shape[0] * (display_width / output_frame.shape[1]))
                        display_frame = cv2.resize(output_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                    cv2.imshow("Person ReID Pipeline - Press 'q' to Quit", display_frame)
                except Exception as e: args.no_display = True; print(f"Error displaying frame: {e}. Display disabled.")

            if video_writer is not None:
                 try: video_writer.write(output_frame) # Luôn lưu frame gốc
                 except Exception as e: print(f"Error writing frame {frame_count}: {e}")

            # --- ĐÃ XÓA LOGIC time.sleep ---

            # --- Xử lý Quit Key ---
            if not args.no_display:
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                      print("Processing stopped by user ('q' pressed).")
                      break

    except KeyboardInterrupt: print("\nProcessing interrupted by user (Ctrl+C).")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); import traceback; traceback.print_exc()
    finally:
        # --- Cleanup ---
        end_time_pipeline = time.time(); total_time = end_time_pipeline - start_time_pipeline # Thời gian chỉ tính từ sau frame 1
        final_processed_count = frame_count - 1 if first_frame_processed else frame_count # Số frame thực sự trong vòng lặp chính
        avg_fps = final_processed_count / total_time if total_time > 1e-6 and final_processed_count > 0 else 0
        print("\n--- Video Processing Summary ---")
        print(f"Processed {final_processed_count} frames (after initial warmup frame).")
        if total_time > 0: print(f"Total Processing Time (excluding first frame): {total_time:.2f} seconds.")
        if avg_fps > 0: print(f"Average Overall FPS: {avg_fps:.2f}") # Bỏ (incl. delay)
        if gallery: print(f"Final Gallery Size: {gallery.get_gallery_size()}")
        if video_writer is not None: print("Releasing video writer..."); video_writer.release()
        if output_path and video_writer is not None: print(f"Output video saved to: {output_path}")
        cv2.destroyAllWindows(); print("Processing finished.")

# --- Block if __name__ == "__main__": để chạy độc lập ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Person ReID Pipeline on Video (Standalone Script)")
    # --- Định nghĩa args như cũ (không cần target-fps) ---
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', '-o', type=str, default='output/main_pipeline_output.mp4', help='Path to save the output video file.')
    parser.add_argument('--reid-config', type=str, default='configs/reid_config.yaml', help='Path to the ReID configuration file.')
    parser.add_argument('--yolo-config', type=str, default='configs/yolo_config.yaml', help='Path to the YOLO/tracking configuration file.')
    parser.add_argument('--no-display', action='store_true', help='Do not display the video window.')
    args = parser.parse_args()

    if not os.path.isfile(args.input): print(f"Error: Input video file not found at '{args.input}'")
    else:
        print("[Standalone Mode] Loading configs and models...")
        try:
            config = load_app_config(args.reid_config, args.yolo_config)
            detector_tracker = DetectorTracker(config['yolo'])
            reid_model = ReIDModel(config['reid'])
            gallery = ReIDGallery(config['reid'])
            print("[Standalone Mode] Models and Gallery loaded.")
            # --- Thêm bước Warmup và xử lý frame đầu tiên cho Standalone Mode ---
            # (Bạn có thể copy khối code xử lý frame đầu tiên từ hàm run_video_processing vào đây nếu muốn
            #  chế độ standalone cũng được warmup, nhưng sẽ làm code dài hơn)
            # --- Hoặc gọi thẳng hàm xử lý chính ---
            run_video_processing(config, detector_tracker, reid_model, gallery, args)
        except ImportError: print("Import Error in Standalone Mode...")
        except Exception as e: print(f"[Standalone Mode] Error: {e}"); import traceback; traceback.print_exc()