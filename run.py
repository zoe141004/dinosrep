# run.py
import argparse
import os
import sys
import time # Dùng để đo thời gian load model
import subprocess # Thêm để gọi streamlink
import torch 
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# --- Đảm bảo các module trong dự án có thể được import ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# -------------------------------------------------------

# --- Import các hàm xử lý tác vụ và các lớp cần thiết ---
try:
    from utils.config_loader import load_app_config
    from models.detection_tracking import DetectorTracker
    from models.reid import ReIDModel
    from utils.gallery import ReIDGallery
except ImportError as e:
    print(f"Error: Failed to import core modules (utils/models). {e}")
    sys.exit(1)

# Import các hàm thực thi tác vụ từ các file tương ứng
try:
    from scripts.process_video_full_pipeline import run_video_processing
except ImportError: run_video_processing = None # Gán None nếu import lỗi
try:
    from scripts.extract_single_feature import run_feature_extraction
except ImportError: run_feature_extraction = None
try:
    from scripts.process_image_folder import run_folder_processing
except ImportError: run_folder_processing = None
try:
    # <<< IMPORT HÀM XỬ LÝ LIVE STREAM MỚI >>>
    from scripts.process_live_stream import run_live_stream_processing
except ImportError: run_live_stream_processing = None
# -----------------------------------------------------

# <<< HÀM HỖ TRỢ LẤY URL TỪ YOUTUBE (Thêm vào run.py) >>>
def get_stream_url_from_youtube(youtube_url):
    """Uses streamlink CLI to get the direct stream URL."""
    print(f"Attempting to get direct stream URL for: {youtube_url} using streamlink...")
    try:
        # Check if streamlink is installed
        try:
            subprocess.run(["streamlink", "--version"], check=True, capture_output=True, text=True, timeout=5) # Add timeout
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print("Error: 'streamlink' command not found or timed out. Please install it (`pip install streamlink`).")
            return None

        # Execute streamlink to get the URL
        process = subprocess.run(
            ["streamlink", "--stream-url", youtube_url, "best"],
            check=True, capture_output=True, text=True, encoding='utf-8', timeout=30 # Add timeout
        )
        stream_url = process.stdout.strip()
        if stream_url and ('http' in stream_url or 'rtmp' in stream_url):
             print(f"Found stream URL: {stream_url}")
             return stream_url
        else:
             print(f"Error: streamlink did not return a valid URL. Output:\n{process.stdout}\n{process.stderr}")
             return None
    except subprocess.TimeoutExpired:
        print("Error: streamlink command timed out.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running streamlink: {e}")
        print(f"Streamlink stderr:\n{e.stderr}")
        print("Is the YouTube URL correct and live? Does it require login?")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while getting stream URL: {e}")
        return None
# <<< KẾT THÚC HÀM HỖ TRỢ >>>


# --- Hàm bao bọc cho từng tác vụ để load model cần thiết ---

def run_video_task(config, args):
    """Loads models for video task and calls the processing function."""
    if run_video_processing is None: print("Error: Video processing function not available."); return
    print("\n--- Initializing Models for VIDEO task ---")
    start_load_time = time.time()
    try:
        # Apply command-line overrides BEFORE initializing models/gallery
        if args.use_re_ranking is not None: config['reid']['use_re_ranking'] = args.use_re_ranking
        # <<< THÊM OVERRIDE CHO FPS VÀ BATCH SIZE >>>
        if hasattr(args, 'target_fps') and args.target_fps is not None: config['reid']['target_fps'] = args.target_fps
        if hasattr(args, 'reid_batch_size') and args.reid_batch_size is not None: config['reid']['reid_batch_size'] = args.reid_batch_size
        # --------------------------------------------

        detector_tracker = DetectorTracker(config['yolo'])
        reid_model = ReIDModel(config['reid'])
        gallery = ReIDGallery(config['reid'])
        print(f"Models Initialized (Load Time: {time.time() - start_load_time:.2f}s)")
        print(f"Effective Settings: ReRank={config['reid']['use_re_ranking']}, TargetFPS={config['reid'].get('target_fps')}, Batch={config['reid'].get('reid_batch_size')}")
         # <<< --- THÊM BƯỚC WARM-UP MODEL ReID --- >>>
        if config['reid'].get('use_torch_compile', False) or True: # Luôn warmup nhẹ nhàng ngay cả khi không compile
            print("\n--- Warming up ReID model ---")
            warmup_start_time = time.time()
            try:
                reid_cfg = config['reid']
                device = reid_cfg['device']
                # Lấy kích thước input và batch size từ config
                input_h, input_w = reid_cfg['reid_input_size'] # [H, W]
                batch_size = reid_cfg.get('reid_batch_size', 16) # Dùng batch size config, hoặc 16 nếu nhỏ
                num_warmup_runs = 5 # Số lần chạy thử

                # Tạo dữ liệu giả (dummy data) - batch ảnh đen hoặc nhiễu
                # Kích thước (batch, channels, height, width)
                dummy_input_batch = torch.zeros((batch_size, 3, input_h, input_w), dtype=torch.float32).to(device)
                # Hoặc dùng ảnh nhiễu: torch.randn((batch_size, 3, input_h, input_w), dtype=torch.float32).to(device)

                print(f"Running {num_warmup_runs} warmup inferences with batch size {batch_size} on device {device}...")
                for _ in range(num_warmup_runs):
                    with torch.no_grad(): # Không cần tính gradient khi warmup
                        # Gọi hàm inference chính (có thể có autocast bên trong)
                         _ = reid_model.extract_features_optimized([dummy_input_batch[i:i+1].cpu().numpy().transpose(0,2,3,1)[0] for i in range(batch_size)]) # Cần chuyển tensor giả -> list numpy giả
                        # Hoặc gọi thẳng model nếu hàm extract phức tạp:
                        # with torch.amp.autocast(device_type='cuda', enabled=reid_cfg.get('use_mixed_precision', False)):
                        #      _ = reid_model.model(dummy_input_batch)


                # Đồng bộ hóa GPU để đảm bảo các tác vụ warmup đã hoàn thành (quan trọng nếu compile)
                if 'cuda' in str(device):
                    torch.cuda.synchronize()

                warmup_time = time.time() - warmup_start_time
                print(f"--- ReID Model Warmup Finished (Time: {warmup_time:.2f} seconds) ---")

            except Exception as e:
                print(f"Warning: ReID model warmup failed: {e}")
                # Vẫn tiếp tục chạy nhưng frame đầu có thể chậm
        # <<< --- KẾT THÚC BƯỚC WARM-UP --- >>>

        print(f"\n--- Running Task: VIDEO ---")
        run_video_processing(config, detector_tracker, reid_model, gallery, args)
    except Exception as e: print(f"Error in video task: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    print(f"--- Task VIDEO Finished ---")

def run_extract_task(config, args):
    """Loads model for extraction task and calls the processing function."""
    if run_feature_extraction is None: print("Error: Feature extraction function not available."); return
    print("\n--- Initializing Models for EXTRACT task ---")
    start_load_time = time.time()
    try:
        reid_model = ReIDModel(config['reid'])
        print(f"ReID Model Initialized (Load Time: {time.time() - start_load_time:.2f}s)")
        run_feature_extraction(config['reid'], reid_model, args)
    except Exception as e: print(f"Error in extract task: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    print(f"--- Task EXTRACT Finished ---")

def run_folder_task(config, args):
    """Loads models for folder processing task and calls the processing function."""
    if run_folder_processing is None: print("Error: Folder processing function not available."); return
    print("\n--- Initializing Models for FOLDER task ---")
    start_load_time = time.time()
    try:
        # Apply command-line overrides
        config['reid']['use_re_ranking'] = args.use_re_ranking # Already handled boolean optional action
        print(f"Folder Task - Re-ranking set to: {config['reid']['use_re_ranking']}")
        # <<< THÊM OVERRIDE CHO FPS VÀ BATCH SIZE (Mặc dù folder task không dùng fps) >>>
        if hasattr(args, 'target_fps') and args.target_fps is not None: config['reid']['target_fps'] = args.target_fps
        if hasattr(args, 'reid_batch_size') and args.reid_batch_size is not None: config['reid']['reid_batch_size'] = args.reid_batch_size
        # ------------------------------------------------------------------------

        reid_model = ReIDModel(config['reid'])
        gallery = ReIDGallery(config['reid'])
        print(f"Models Initialized (Load Time: {time.time() - start_load_time:.2f}s)")
        run_folder_processing(config['reid'], reid_model, gallery, args)
    except Exception as e: print(f"Error in folder task: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    print(f"--- Task FOLDER Finished ---")


# <<< HÀM WRAPPER MỚI CHO LIVE TASK >>>
def run_live_task(config, args):
    """Loads models for live stream task and calls the processing function."""
    if run_live_stream_processing is None:
        print("Error: Live stream processing function not available.")
        return

    stream_url_to_use = args.stream_url
    # --- Tự động lấy URL stream nếu input là link YouTube ---
    if args.stream_url and ('https://www.youtube.com/watch?v=...7' in args.stream_url or 'https://www.youtube.com/watch?v=...8' in args.stream_url):
         direct_url = get_stream_url_from_youtube(args.stream_url)
         if direct_url:
              stream_url_to_use = direct_url
         else:
              print("Failed to get direct stream URL from YouTube link. Exiting.")
              return # Thoát nếu không lấy được link

    if not stream_url_to_use:
         print("Error: No valid stream URL provided or obtained.")
         return

    # --- Gán lại stream url đã được xử lý vào args để hàm processing sử dụng ---
    args.stream_url = stream_url_to_use # Quan trọng: cập nhật args để truyền vào hàm xử lý

    print("\n--- Initializing Models for LIVE task ---")
    start_load_time = time.time()
    try:
        # Apply command-line overrides to config
        if args.use_re_ranking is not None: config['reid']['use_re_ranking'] = args.use_re_ranking
        if args.target_fps is not None: config['reid']['target_fps'] = args.target_fps
        if args.reid_batch_size is not None: config['reid']['reid_batch_size'] = args.reid_batch_size

        detector_tracker = DetectorTracker(config['yolo'])
        reid_model = ReIDModel(config['reid'])
        gallery = ReIDGallery(config['reid'])
        print(f"Models Initialized (Load Time: {time.time() - start_load_time:.2f}s)")
        print(f"Effective Settings: ReRank={config['reid']['use_re_ranking']}, TargetFPS={config['reid'].get('target_fps')}, Batch={config['reid'].get('reid_batch_size')}")

        # <<< --- THÊM BƯỚC WARM-UP MODEL ReID (Giống hệt phần video) --- >>>
        if config['reid'].get('use_torch_compile', False) or True:
            print("\n--- Warming up ReID model ---")
            warmup_start_time = time.time()
            try:
                reid_cfg = config['reid']
                device = reid_cfg['device']
                input_h, input_w = reid_cfg['reid_input_size']
                batch_size = reid_cfg.get('reid_batch_size', 16)
                num_warmup_runs = 5
                dummy_input_batch = torch.zeros((batch_size, 3, input_h, input_w), dtype=torch.float32).to(device)
                print(f"Running {num_warmup_runs} warmup inferences with batch size {batch_size} on device {device}...")
                for _ in range(num_warmup_runs):
                    with torch.no_grad():
                        # Gọi hàm inference chính để warmup cả tiền xử lý bên trong nó nếu có
                        # Cần tạo list numpy giả từ tensor dummy
                        dummy_np_list = [dummy_input_batch[i:i+1].cpu().numpy().transpose(0,2,3,1)[0] for i in range(batch_size)]
                        _ = reid_model.extract_features_optimized(dummy_np_list)

                if 'cuda' in str(device): torch.cuda.synchronize()
                warmup_time = time.time() - warmup_start_time
                print(f"--- ReID Model Warmup Finished (Time: {warmup_time:.2f} seconds) ---")
            except Exception as e:
                print(f"Warning: ReID model warmup failed: {e}")
        # <<< --- KẾT THÚC BƯỚC WARM-UP --- >>>
        print(f"\n--- Running Task: LIVE ---")
        # Gọi hàm xử lý live stream từ scripts/process_live_stream.py
        run_live_stream_processing(config, detector_tracker, reid_model, gallery, args)
    except Exception as e:
        print(f"Error in live task: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print(f"--- Task LIVE Finished ---")
# <<< KẾT THÚC HÀM WRAPPER MỚI >>>


def main():
    # --- Định nghĩa Argument Parser chính ---
    parser = argparse.ArgumentParser(
        description="Person Re-Identification Project Runner.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--reid-config', type=str, default='configs/reid_config.yaml', help='Path to ReID config (default: %(default)s).')
    parser.add_argument('--yolo-config', type=str, default='configs/yolo_config.yaml', help='Path to YOLO config (default: %(default)s).')

    subparsers = parser.add_subparsers(dest='task', required=True, title='Available tasks')

    # --- Subcommand: video ---
    parser_video = subparsers.add_parser('video', help='Process a video file.')
    parser_video.add_argument('--input', '-i', type=str, required=True, help='Input video file path.')
    parser_video.add_argument('--output', '-o', type=str, default='output/run_video_output.mp4', help='Output video file path (default: %(default)s).')
    parser_video.add_argument('--no-display', action='store_true', help='Disable display window.')
    parser_video.add_argument('--use-re-ranking', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable re-ranking (overrides config).')
    # <<< THÊM ARGS CHO VIDEO >>>
    parser_video.add_argument('--target-fps', type=float, default=None, help='Attempt to limit processing FPS (overrides config).')
    parser_video.add_argument('--reid-batch-size', type=int, default=None, help='ReID inference batch size (overrides config).')
    # --------------------------
    parser_video.set_defaults(func=run_video_task)

    # --- Subcommand: extract ---
    parser_extract = subparsers.add_parser('extract', help='Extract feature from a single image.')
    parser_extract.add_argument('--image', '-i', type=str, required=True, help='Input image file path.')
    parser_extract.add_argument('--output', '-o', type=str, default=None, help='Output feature file path (.pt or .npy) (optional).')
    parser_extract.add_argument('--show', action='store_true', help='Display input image.')
    parser_extract.set_defaults(func=run_extract_task)

    # --- Subcommand: folder ---
    parser_folder = subparsers.add_parser('folder', help='Process a folder of cropped images.')
    parser_folder.add_argument('--input-folder', '-if', type=str, required=True, help='Input folder path.')
    parser_folder.add_argument('--output-folder', '-of', type=str, default='output/run_grouped_output', help='Output folder for grouped images (default: %(default)s).')
    parser_folder.add_argument('--use-re-ranking', action=argparse.BooleanOptionalAction, default=True, help='Enable re-ranking (default: True). Use --no-re-ranking to disable.')
    parser_folder.add_argument('--group-output', action=argparse.BooleanOptionalAction, default=True, help='Group output images by ID (default: True). Use --no-group-output to disable.')
    parser_folder.add_argument('--zip-output', action='store_true', help='Zip the output folder after grouping.')
    parser_folder.add_argument('--zip-filename', type=str, default='run_grouped_images', help='Base name for output zip file (default: %(default)s).')
    parser_folder.add_argument('--no-display', action='store_true', help='Disable image display.')
    # <<< THÊM ARGS CHO FOLDER (Mặc dù không dùng fps nhưng để nhất quán) >>>
    parser_folder.add_argument('--target-fps', type=float, default=None, help='(Not used by folder task) Attempt to limit processing FPS.')
    parser_folder.add_argument('--reid-batch-size', type=int, default=None, help='ReID inference batch size (overrides config).')
    # -------------------------------------------------------------------
    parser_folder.set_defaults(func=run_folder_task)

    # --- Subcommand: live --- <<< SUBCOMMAND MỚI >>>
    parser_live = subparsers.add_parser('live', help='Process a live video stream (e.g., YouTube, HLS, RTSP).')
    parser_live.add_argument('--stream-url', '-s', type=str, required=True, help='Direct stream URL OR a YouTube Live page URL.')
    parser_live.add_argument('--output', '-o', type=str, default=None, help='Output video file path (optional).')
    parser_live.add_argument('--no-display', action='store_true', help='Disable display window.')
    parser_live.add_argument('--use-re-ranking', action=argparse.BooleanOptionalAction, default=False, help='Enable/disable re-ranking (default: False).') # Mặc định False cho live
    parser_live.add_argument('--target-fps', type=float, default=None, help='Attempt to limit processing FPS (overrides config).')
    parser_live.add_argument('--reid-batch-size', type=int, default=None, help='ReID inference batch size (overrides config).')
    parser_live.set_defaults(func=run_live_task) # Gọi hàm run_live_task


    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Load Configs ---
    print("--- Loading Configurations ---")
    try:
        config = load_app_config(args.reid_config, args.yolo_config)
    except FileNotFoundError as e: print(f"Error: {e}"); sys.exit(1)
    except Exception as e: print(f"Error loading configurations: {e}"); sys.exit(1)
    print("--- Configurations Loaded ---")

    # --- Execute the selected task's function ---
    args.func(config, args) # Gọi hàm được gán bởi set_defaults

if __name__ == "__main__":
    main()