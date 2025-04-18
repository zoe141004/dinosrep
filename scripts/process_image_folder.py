# scripts/process_image_folder.py
import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import shutil
import sys
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# --- Add project root to sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# ------------------------------------

# Imports for standalone mode
try:
    from utils.config_loader import load_app_config
    from utils.gallery import ReIDGallery
    from models.reid import ReIDModel
    # from utils.visualization import draw_single_bbox # Optional for drawing during processing
except ImportError:
    if __name__ == '__main__':
        print("Error: Could not import project modules in standalone mode.")
        print("Ensure script is run from project root or PYTHONPATH is set.")
        sys.exit(1)
    else:
        pass

# Global mapping for this script (cleared each run)
image_id_mapping = defaultdict(list)

def show_image(image_path, title="Image"):
    """Displays an image using matplotlib."""
    try:
        img = Image.open(image_path)
        plt.figure(figsize=(5, 8)) # Adjust figure size
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False) # Show non-blocking
        plt.pause(0.1) # Shorter pause
    except FileNotFoundError:
        print(f"Cannot display. Image not found: {image_path}")
    except ImportError:
         print("Error: Cannot display image. Matplotlib or Pillow might be missing.")
    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! HÀM CHÍNH ĐỂ XỬ LÝ FOLDER (ĐƯỢC GỌI BỞI run.py) !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def run_folder_processing(reid_config, reid_model, gallery, args):
    """
    Processes images in a folder for ReID using pre-initialized models/gallery.

    Args:
        reid_config (dict): The 'reid' section of the configuration (potentially updated by run.py).
        reid_model (ReIDModel): Initialized ReID model instance.
        gallery (ReIDGallery): Initialized ReID gallery instance.
        args (argparse.Namespace): Parsed arguments specific to the folder task.
    """
    global image_id_mapping
    image_id_mapping.clear() # Reset mapping for this run

    input_folder = args.input_folder
    output_folder = args.output_folder
    should_group_output = args.group_output
    should_zip_output = args.zip_output
    zip_filename_base = args.zip_filename
    no_display = args.no_display

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return

    print(f"Processing images in: {input_folder} (using pre-loaded models)")
    print(f" - Re-ranking: {reid_config['use_re_ranking']}") # Show effective setting
    print(f" - Grouping Output: {should_group_output}")
    if should_group_output: print(f" - Output Folder: {output_folder}")
    if should_zip_output: print(f" - Zip Output: Enabled")


    try:
         image_files = sorted([f for f in os.listdir(input_folder)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
         if not image_files:
              print(f"Warning: No image files found in {input_folder}")
              return
    except OSError as e:
         print(f"Error reading input folder {input_folder}: {e}")
         return


    start_time = time.time()
    processed_count = 0
    error_count = 0
    mi_time = 100
    ma_time = 0
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(input_folder, img_file)
        # print(f"\n--- Processing Image {i+1}/{len(image_files)}: {img_file} ---") # Verbose log

        # --- 1. Extract Feature ---
        time1 = time.time()
        query_feature = reid_model.extract_feature_single(img_path)
        extract_time = time.time() - time1
        mi_time = min(extract_time,mi_time)
        ma_time = max(extract_time,ma_time)

        if query_feature is None:
            print(f"Skipping image {img_file} due to feature extraction error.")
            error_count += 1
            continue

        # --- 2. Compare with Gallery and Assign ID ---
        # Gallery's assign_ids handles comparison, new ID logic, and internal update
        assigned_reid_ids = gallery.assign_ids(query_feature) # Input needs batch dim

        if not assigned_reid_ids or assigned_reid_ids[0] == -1:
             print(f"Error: Could not assign valid ID for {img_file}")
             error_count += 1
             continue

        assigned_id = assigned_reid_ids[0]
        processed_count += 1

        # --- 3. Update Mapping and Display (Optional) ---
        image_id_mapping[assigned_id].append(img_path)
        # print(f"Assigned ID: {assigned_id} to {img_file}") # Verbose log

        if not no_display:
            status = f"ID: {assigned_id}"
            show_image(img_path, f"{img_file}\n({status})")

    total_time = time.time() - start_time
    print(f"\n--- Folder Processing Summary ---")
    print(f"Min time extract:{mi_time}")
    print(f"Max time extract:{ma_time}")
    print(f"Processed {processed_count} images ({error_count} errors) in {total_time:.2f} seconds.")
    if gallery: print(f"Total unique IDs assigned/found: {gallery.get_gallery_size()}")


    # --- 4. Group Images by ID ---
    if should_group_output and processed_count > 0:
        print(f"\nGrouping images by ID into: {output_folder}")
        if os.path.exists(output_folder):
            print(f"Warning: Output folder '{output_folder}' already exists. Files might be overwritten.")
        else:
            try:
                os.makedirs(output_folder)
            except OSError as e:
                print(f"Error creating output folder {output_folder}: {e}. Skipping grouping.")
                should_group_output = False # Disable grouping if folder fails

        if should_group_output: # Check again if directory creation was successful
            copy_errors = 0
            for pid, paths in image_id_mapping.items():
                pid_folder = os.path.join(output_folder, f"ID_{pid}")
                try:
                     os.makedirs(pid_folder, exist_ok=True) # Create subdir for ID
                     for img_path in paths:
                         img_name = os.path.basename(img_path)
                         target_path = os.path.join(pid_folder, img_name)
                         shutil.copy2(img_path, target_path) # copy2 preserves metadata
                except OSError as e:
                     print(f"Error creating directory or copying for ID {pid}: {e}")
                     copy_errors += len(paths) # Count all images for that ID as errors
                except Exception as e:
                     print(f"Error copying file {img_path} for ID {pid}: {e}")
                     copy_errors += 1
            if copy_errors == 0:
                 print(f"Images grouped successfully.")
            else:
                 print(f"Images grouped with {copy_errors} errors.")


    # --- 5. Zip Output (Optional) ---
    if should_group_output and should_zip_output and processed_count > 0:
        if not os.path.isdir(output_folder):
             print("Cannot zip output: Grouping folder does not exist or failed.")
        else:
             # Put zip file one level up from the output folder
             zip_path_base = os.path.join(os.path.dirname(output_folder) or '.', zip_filename_base)
             print(f"Zipping output folder '{output_folder}' to '{zip_path_base}.zip' ...")
             try:
                 shutil.make_archive(zip_path_base, 'zip', output_folder)
                 print(f"✅ Folder zipped successfully.")
             except Exception as e:
                 print(f"Error zipping output folder: {e}")

    if not no_display:
         print("Closing any open plot windows...")
         plt.close('all')


# ==============================================================
#     BLOCK ĐỂ CHẠY FILE NÀY ĐỘC LẬP (Standalone Mode)
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image folder for ReID (Standalone Script)")
    # --- Copy argparse definitions from run.py's folder subcommand ---
    parser.add_argument('--input-folder', '-if', type=str, required=True, help='Path to the folder containing cropped person images.')
    parser.add_argument('--output-folder', '-of', type=str, default='output/standalone_grouped_output', help='Folder to save grouped images.')
    parser.add_argument('--reid-config', type=str, default='configs/reid_config.yaml', help='Path to the ReID configuration file.')
    parser.add_argument('--use-re-ranking', action=argparse.BooleanOptionalAction, default=True, help='Enable re-ranking (default). Use --no-re-ranking to disable.')
    parser.add_argument('--group-output', action=argparse.BooleanOptionalAction, default=True, help='Group output images by ID (default). Use --no-group-output to disable.')
    parser.add_argument('--zip-output', action='store_true', help='Zip the output folder after grouping.')
    parser.add_argument('--zip-filename', type=str, default='standalone_grouped_images', help='Base name for the output zip file.')
    parser.add_argument('--no-display', action='store_true', help='Do not display images during processing.')
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder not found at '{args.input_folder}'")
    else:
        print("[Standalone Mode] Loading config and models...")
        try:
            # Load config
            config = load_app_config(reid_config_path=args.reid_config)
            # Apply re-ranking setting from args to config BEFORE initializing models/gallery
            config['reid']['use_re_ranking'] = args.use_re_ranking
            print(f"[Standalone Mode] Re-ranking set to: {config['reid']['use_re_ranking']}")

            # Initialize models
            reid_model = ReIDModel(config['reid'])
            gallery = ReIDGallery(config['reid'])
            print("[Standalone Mode] Models loaded.")
            # Call the main logic function
            run_folder_processing(config['reid'], reid_model, gallery, args)
        except ImportError:
             # Error already handled by try-except block at top for standalone mode
             pass
        except (FileNotFoundError, RuntimeError, ValueError, Exception) as e:
            print(f"[Standalone Mode] Error during initialization or processing: {e}")
            import traceback
            traceback.print_exc()