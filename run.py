# run.py
import argparse
import os
import sys
import time
import subprocess

# --- Add project root to sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# ------------------------------------

# --- Import core modules ---
try:
    from utils.config_loader import load_app_config
    from models.detection_tracking import DetectorTracker
    from models.reid import ReIDModel
    from utils.gallery import ReIDGallery
except ImportError as e: print(f"Error: Failed to import core modules. {e}"); sys.exit(1)

# --- Import task functions ---
try: from scripts.process_video_full_pipeline import run_video_processing
except ImportError: run_video_processing = None
try: from scripts.extract_single_feature import run_feature_extraction
except ImportError: run_feature_extraction = None
try: from scripts.process_image_folder import run_folder_processing
except ImportError: run_folder_processing = None
try: from scripts.process_live_stream import run_live_stream_processing
except ImportError: run_live_stream_processing = None
# ---------------------------

# --- YouTube Stream URL Helper ---
def get_stream_url_from_youtube(youtube_url):
    """Uses streamlink CLI to get the direct stream URL."""
    print(f"Attempting to get direct stream URL for: {youtube_url} using streamlink...")
    try:
        try: subprocess.run(["streamlink", "--version"], check=True, capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
             print("Error: 'streamlink' not found or timed out. Install with `pip install streamlink`."); return None
        process = subprocess.run(["streamlink", "--stream-url", youtube_url, "best"], check=True, capture_output=True, text=True, encoding='utf-8', timeout=30)
        stream_url = process.stdout.strip()
        if stream_url and ('http' in stream_url or 'rtmp' in stream_url): print(f"Found stream URL: {stream_url}"); return stream_url
        else: print(f"Error: streamlink did not return a valid URL. Output:\n{process.stdout}\n{process.stderr}"); return None
    except subprocess.TimeoutExpired: print("Error: streamlink command timed out."); return None
    except subprocess.CalledProcessError as e: print(f"Error running streamlink: {e}\nstderr: {e.stderr}"); return None
    except Exception as e: print(f"An unexpected error occurred while getting stream URL: {e}"); return None
# ---------------------------------

# --- Task Wrapper Functions ---
def run_video_task(config, args):
    if run_video_processing is None: print("Error: Video processing function not available."); return
    print("\n--- Initializing Models for VIDEO task ---")
    start_load_time = time.time()
    try:
        # Apply command-line overrides (trừ target_fps)
        if args.use_re_ranking is not None: config['reid']['use_re_ranking'] = args.use_re_ranking
        if args.reid_batch_size is not None: config['reid']['reid_batch_size'] = args.reid_batch_size

        detector_tracker = DetectorTracker(config['yolo'])
        reid_model = ReIDModel(config['reid'])
        gallery = ReIDGallery(config['reid'])
        print(f"Models Initialized (Load Time: {time.time() - start_load_time:.2f}s)")
        print(f"Effective Settings: ReRank={config['reid']['use_re_ranking']}, Batch={config['reid'].get('reid_batch_size')}") # Bỏ TargetFPS
        run_video_processing(config, detector_tracker, reid_model, gallery, args)
    except Exception as e: print(f"Error in video task: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    print(f"--- Task VIDEO Finished ---")

def run_extract_task(config, args):
    if run_feature_extraction is None: print("Error: Feature extraction function not available."); return
    print("\n--- Initializing Models for EXTRACT task ---"); start_load_time = time.time()
    try:
        reid_model = ReIDModel(config['reid'])
        print(f"ReID Model Initialized (Load Time: {time.time() - start_load_time:.2f}s)")
        run_feature_extraction(config['reid'], reid_model, args)
    except Exception as e: print(f"Error in extract task: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    print(f"--- Task EXTRACT Finished ---")

def run_folder_task(config, args):
    if run_folder_processing is None: print("Error: Folder processing function not available."); return
    print("\n--- Initializing Models for FOLDER task ---"); start_load_time = time.time()
    try:
        # Apply command-line overrides (trừ target_fps)
        config['reid']['use_re_ranking'] = args.use_re_ranking
        if args.reid_batch_size is not None: config['reid']['reid_batch_size'] = args.reid_batch_size
        print(f"Folder Task - Re-ranking: {config['reid']['use_re_ranking']}, Batch: {config['reid'].get('reid_batch_size')}")

        reid_model = ReIDModel(config['reid'])
        gallery = ReIDGallery(config['reid'])
        print(f"Models Initialized (Load Time: {time.time() - start_load_time:.2f}s)")
        run_folder_processing(config['reid'], reid_model, gallery, args)
    except Exception as e: print(f"Error in folder task: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    print(f"--- Task FOLDER Finished ---")

def run_live_task(config, args):
    if run_live_stream_processing is None: print("Error: Live stream processing function not available."); return
    stream_url_to_use = args.stream_url
    if args.stream_url and ('https://youtu.be/...9' in args.stream_url or 'https://www.youtube.com/live/...0' in args.stream_url):
         direct_url = get_stream_url_from_youtube(args.stream_url)
         if direct_url: stream_url_to_use = direct_url
         else: print("Failed to get direct stream URL from YouTube link. Exiting."); return
    if not stream_url_to_use: print("Error: No valid stream URL provided or obtained."); return
    args.stream_url = stream_url_to_use

    print("\n--- Initializing Models for LIVE task ---"); start_load_time = time.time()
    try:
        # Apply command-line overrides (trừ target_fps)
        if args.use_re_ranking is not None: config['reid']['use_re_ranking'] = args.use_re_ranking
        if args.reid_batch_size is not None: config['reid']['reid_batch_size'] = args.reid_batch_size

        detector_tracker = DetectorTracker(config['yolo'])
        reid_model = ReIDModel(config['reid'])
        gallery = ReIDGallery(config['reid'])
        print(f"Models Initialized (Load Time: {time.time() - start_load_time:.2f}s)")
        print(f"Effective Settings: ReRank={config['reid']['use_re_ranking']}, Batch={config['reid'].get('reid_batch_size')}") # Bỏ TargetFPS
        run_live_stream_processing(config, detector_tracker, reid_model, gallery, args)
    except Exception as e: print(f"Error in live task: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    print(f"--- Task LIVE Finished ---")
# --------------------------------

def main():
    parser = argparse.ArgumentParser(description="Person Re-Identification Project Runner.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--reid-config', type=str, default='configs/reid_config.yaml', help='Path to ReID config (default: %(default)s).')
    parser.add_argument('--yolo-config', type=str, default='configs/yolo_config.yaml', help='Path to YOLO config (default: %(default)s).')
    subparsers = parser.add_subparsers(dest='task', required=True, title='Available tasks')

    # --- Subcommand: video ---
    parser_video = subparsers.add_parser('video', help='Process a video file.')
    parser_video.add_argument('--input', '-i', type=str, required=True, help='Input video file path.')
    parser_video.add_argument('--output', '-o', type=str, default='output/run_video_output.mp4', help='Output video file path (default: %(default)s).')
    parser_video.add_argument('--no-display', action='store_true', help='Disable display window.')
    parser_video.add_argument('--use-re-ranking', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable re-ranking (overrides config).')
    parser_video.add_argument('--reid-batch-size', type=int, default=None, help='ReID inference batch size (overrides config).')
    # --- ĐÃ XÓA --target-fps ---
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
    parser_folder.add_argument('--use-re-ranking', action=argparse.BooleanOptionalAction, default=True, help='Enable re-ranking (default: True).')
    parser_folder.add_argument('--group-output', action=argparse.BooleanOptionalAction, default=True, help='Group output images by ID (default: True).')
    parser_folder.add_argument('--zip-output', action='store_true', help='Zip the output folder.')
    parser_folder.add_argument('--zip-filename', type=str, default='run_grouped_images', help='Base name for output zip file (default: %(default)s).')
    parser_folder.add_argument('--no-display', action='store_true', help='Disable image display.')
    parser_folder.add_argument('--reid-batch-size', type=int, default=None, help='ReID inference batch size (overrides config).')
    # --- ĐÃ XÓA --target-fps ---
    parser_folder.set_defaults(func=run_folder_task)

    # --- Subcommand: live ---
    parser_live = subparsers.add_parser('live', help='Process a live video stream.')
    parser_live.add_argument('--stream-url', '-s', type=str, required=True, help='Direct stream URL OR a YouTube Live page URL.')
    parser_live.add_argument('--output', '-o', type=str, default=None, help='Output video file path (optional).')
    parser_live.add_argument('--no-display', action='store_true', help='Disable display window.')
    parser_live.add_argument('--use-re-ranking', action=argparse.BooleanOptionalAction, default=False, help='Enable/disable re-ranking (default: False).')
    parser_live.add_argument('--reid-batch-size', type=int, default=None, help='ReID inference batch size (overrides config).')
    # --- ĐÃ XÓA --target-fps ---
    parser_live.set_defaults(func=run_live_task)

    # --- Parse Arguments & Execute Task ---
    args = parser.parse_args()
    print("--- Loading Configurations ---")
    try: config = load_app_config(args.reid_config, args.yolo_config)
    except Exception as e: print(f"Error loading configurations: {e}"); sys.exit(1)
    print("--- Configurations Loaded ---")
    args.func(config, args) # Execute the task function

if __name__ == "__main__":
    main()