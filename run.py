# run.py
import argparse
import os
import sys
import time # Dùng để đo thời gian load model

# --- Đảm bảo các module trong dự án có thể được import ---
# Thêm thư mục gốc của dự án vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# -------------------------------------------------------

# --- Import các hàm xử lý tác vụ và các lớp cần thiết ---
# Tách riêng việc import để xử lý lỗi rõ ràng hơn
try:
    from utils.config_loader import load_app_config
    from models.detection_tracking import DetectorTracker
    from models.reid import ReIDModel
    from utils.gallery import ReIDGallery
except ImportError as e:
    print(f"Error: Failed to import core modules (utils/models). {e}")
    print("Please ensure the project structure is correct and all dependencies are installed.")
    sys.exit(1)

# Import các hàm thực thi tác vụ từ các file tương ứng
try:
    from scripts.process_video_full_pipeline import run_video_processing
except ImportError:
    def run_video_processing(*args, **kwargs):
        print("Error: Could not import 'run_video_processing' from main_pipeline.py")
        sys.exit(1)

try:
    from scripts.extract_single_feature import run_feature_extraction
except ImportError:
    def run_feature_extraction(*args, **kwargs):
        print("Error: Could not import 'run_feature_extraction' from scripts/extract_single_feature.py")
        sys.exit(1)

try:
    from scripts.process_image_folder import run_folder_processing
except ImportError:
    def run_folder_processing(*args, **kwargs):
        print("Error: Could not import 'run_folder_processing' from scripts/process_image_folder.py")
        sys.exit(1)
# -----------------------------------------------------


def main():
    # --- Định nghĩa Argument Parser chính ---
    parser = argparse.ArgumentParser(
        description="Person Re-Identification Project Runner. Choose a task (video, extract, folder) to perform.",
        formatter_class=argparse.RawTextHelpFormatter # Giữ định dạng help message tốt hơn
    )

    # --- Đối số Toàn cục ---
    # Các đối số này áp dụng cho tất cả các tác vụ (subcommand)
    parser.add_argument('--reid-config', type=str, default='configs/reid_config.yaml',
                        help='Path to the ReID configuration file (default: %(default)s).')
    parser.add_argument('--yolo-config', type=str, default='configs/yolo_config.yaml',
                        help='Path to the YOLO/tracking configuration file (default: %(default)s).')

    # --- Định nghĩa Subcommand Parser ---
    subparsers = parser.add_subparsers(dest='task', required=True,
                                       title='Available tasks',
                                       description='Choose one of the following tasks to run:',
                                       help='Task specific arguments:')

    # --- Subcommand: video ---
    parser_video = subparsers.add_parser('video', help='Run detection, tracking, and ReID on a video file.')
    parser_video.add_argument('--input', '-i', type=str, required=True, help='Path to the input video file.')
    parser_video.add_argument('--output', '-o', type=str, default='output/run_video_output.mp4', help='Path to save the output video file (default: %(default)s).')
    parser_video.add_argument('--no-display', action='store_true', help='Do not display the video window during processing.')
    # Đặt hàm xử lý mặc định cho subcommand này
    parser_video.set_defaults(func=run_video_task)

    # --- Subcommand: extract ---
    parser_extract = subparsers.add_parser('extract', help='Extract ReID feature from a single image.')
    parser_extract.add_argument('--image', '-i', type=str, required=True, help='Path to the input image file.')
    parser_extract.add_argument('--output', '-o', type=str, default=None, help='Path to save the extracted feature vector (.pt or .npy) (optional).')
    parser_extract.add_argument('--show', action='store_true', help='Display the input image after processing.')
    parser_extract.set_defaults(func=run_extract_task)

    # --- Subcommand: folder ---
    parser_folder = subparsers.add_parser('folder', help='Process a folder of cropped images for ReID and optionally group them.')
    parser_folder.add_argument('--input-folder', '-if', type=str, required=True, help='Path to the folder containing cropped person images.')
    parser_folder.add_argument('--output-folder', '-of', type=str, default='output/run_grouped_output', help='Folder to save grouped images if grouping is enabled (default: %(default)s).')
    # Sử dụng BooleanOptionalAction cho các cờ bật/tắt
    parser_folder.add_argument('--use-re-ranking', action=argparse.BooleanOptionalAction, default=True, help='Enable re-ranking (default). Use --no-re-ranking to disable.')
    parser_folder.add_argument('--group-output', action=argparse.BooleanOptionalAction, default=True, help='Group output images by ID (default). Use --no-group-output to disable.')
    parser_folder.add_argument('--zip-output', action='store_true', help='Zip the output folder after grouping (only if grouping is enabled).')
    parser_folder.add_argument('--zip-filename', type=str, default='run_grouped_images', help='Base name for the output zip file (without .zip) (default: %(default)s).')
    parser_folder.add_argument('--no-display', action='store_true', help='Do not display images during processing.')
    parser_folder.set_defaults(func=run_folder_task)


    # --- Parse Tất cả Arguments ---
    args = parser.parse_args()

    # --- Load Configs (Chung cho tất cả tác vụ) ---
    print("--- Loading Configurations ---")
    try:
        # Luôn load cả hai config, các tác vụ sau sẽ chỉ dùng phần cần thiết
        config = load_app_config(args.reid_config, args.yolo_config)
    except FileNotFoundError as e:
         print(f"Error: Configuration file not found. {e}")
         sys.exit(1)
    except Exception as e:
         print(f"Error loading configurations: {e}")
         sys.exit(1)
    print("--- Configurations Loaded ---")


    # --- Gọi hàm xử lý tương ứng với subcommand đã chọn ---
    # Hàm func được gán bởi set_defaults cho từng subcommand parser
    args.func(config, args)


# --- Hàm bao bọc cho từng tác vụ để load model cần thiết ---

def run_video_task(config, args):
    """Loads models for video task and calls the processing function."""
    print("\n--- Initializing Models for VIDEO task (This might take a while) ---")
    start_load_time = time.time()
    try:
        detector_tracker = DetectorTracker(config['yolo'])
        reid_model = ReIDModel(config['reid'])
        gallery = ReIDGallery(config['reid'])
        load_time = time.time() - start_load_time
        print(f"--- Models Initialized (Load Time: {load_time:.2f} seconds) ---")

        print(f"\n--- Running Task: VIDEO ---")
        # Gọi hàm xử lý video từ main_pipeline.py
        run_video_processing(config, detector_tracker, reid_model, gallery, args)

    except (ImportError, FileNotFoundError, RuntimeError, ValueError, Exception) as e:
        print(f"\nError during model initialization or video processing: {e}")
        import traceback
        traceback.print_exc() # In chi tiết lỗi
        sys.exit(1)
    print(f"--- Task VIDEO Finished ---")


def run_extract_task(config, args):
    """Loads model for extraction task and calls the processing function."""
    print("\n--- Initializing Models for EXTRACT task (This might take a while) ---")
    start_load_time = time.time()
    try:
        # Chỉ cần ReID model cho tác vụ này
        reid_model = ReIDModel(config['reid'])
        load_time = time.time() - start_load_time
        print(f"--- ReID Model Initialized (Load Time: {load_time:.2f} seconds) ---")

        print(f"\n--- Running Task: EXTRACT ---")
        # Gọi hàm xử lý extract từ scripts/extract_single_feature.py
        run_feature_extraction(config['reid'], reid_model, args)

    except (ImportError, FileNotFoundError, RuntimeError, ValueError, Exception) as e:
        print(f"\nError during model initialization or feature extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print(f"--- Task EXTRACT Finished ---")


def run_folder_task(config, args):
    """Loads models for folder processing task and calls the processing function."""
    print("\n--- Initializing Models for FOLDER task (This might take a while) ---")
    start_load_time = time.time()
    try:
        # Cần ReID model và Gallery
        # Điều chỉnh config reid dựa trên args *trước khi* tạo model/gallery
        config['reid']['use_re_ranking'] = args.use_re_ranking
        print(f"Folder Task - Re-ranking set to: {config['reid']['use_re_ranking']}")

        reid_model = ReIDModel(config['reid'])
        gallery = ReIDGallery(config['reid']) # Gallery sẽ dùng config đã cập nhật
        load_time = time.time() - start_load_time
        print(f"--- Models Initialized (Load Time: {load_time:.2f} seconds) ---")

        print(f"\n--- Running Task: FOLDER ---")
        # Gọi hàm xử lý folder từ scripts/process_image_folder.py
        run_folder_processing(config['reid'], reid_model, gallery, args)

    except (ImportError, FileNotFoundError, RuntimeError, ValueError, Exception) as e:
        print(f"\nError during model initialization or folder processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print(f"--- Task FOLDER Finished ---")


if __name__ == "__main__":
    main()