import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import shutil

# Import project modules (adjust paths if running script from root)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add project root

from utils.config_loader import load_app_config
from utils.gallery import ReIDGallery, get_unique_id # Need get_unique_id here too
from utils.reranking import perform_re_ranking, is_reranking_available
from models.reid import ReIDModel
from utils.visualization import draw_single_bbox # For visualization during processing

# Global mapping for this script
image_id_mapping = defaultdict(list) # Map assigned ID -> list of image paths

def show_image(image_path, title="Image"):
    """Displays an image using matplotlib."""
    try:
        img = Image.open(image_path)
        plt.figure() # Create a new figure for each image
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show(block=False) # Show non-blocking
        plt.pause(0.5) # Pause briefly to allow display
    except FileNotFoundError:
        print(f"Cannot display. Image not found: {image_path}")
    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")

def process_folder(args):
    """Processes images in a folder for ReID and groups them."""
    global image_id_mapping, next_person_id

    # 1. Load Config
    config = load_app_config(args.reid_config) # Only need ReID config
    reid_cfg = config['reid']

    # --- Override specific config for folder processing ---
    # Usually want re-ranking enabled for higher accuracy on static images
    reid_cfg['use_re_ranking'] = args.use_re_ranking
    print(f"Folder processing - Use Re-ranking: {reid_cfg['use_re_ranking']}")

    # 2. Initialize ReID Model and Gallery
    print("Initializing ReID model...")
    reid_model = ReIDModel(reid_cfg)
    # For folder processing, the tensor gallery might still be faster if many images
    # Let's stick with the tensor gallery for consistency, but list could also work
    gallery = ReIDGallery(reid_cfg)
    print("Model and Gallery initialized.")


    # 3. Process Images in Folder
    input_folder = args.input_folder
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return

    print(f"Processing images in: {input_folder}")
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])

    start_time = time.time()
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(input_folder, img_file)
        print(f"\n--- Processing Image {i+1}/{len(image_files)}: {img_file} ---")

        # --- 3.1: Extract Feature ---
        # Use the single image extraction method
        query_feature = reid_model.extract_feature_single(img_path)

        if query_feature is None:
            print(f"Skipping image {img_file} due to feature extraction error.")
            continue

        # --- 3.2: Compare with Gallery ---
        # compare_and_update takes a batch, so unsqueeze
        assigned_reid_ids = gallery.compare_and_update(query_feature) # Returns a list of 1 ID

        if not assigned_reid_ids:
             print(f"Error: Could not assign ID for {img_file}")
             continue

        assigned_id = assigned_reid_ids[0]

        # --- 3.3: Update Mapping and Display (Optional) ---
        image_id_mapping[assigned_id].append(img_path)
        print(f"Assigned ID: {assigned_id} to {img_file}")

        if not args.no_display:
            status = "New ID" if len(image_id_mapping[assigned_id]) == 1 else f"Matched ID {assigned_id}"
            show_image(img_path, f"{img_file} ({status})")


    total_time = time.time() - start_time
    print(f"\n--- Folder Processing Summary ---")
    print(f"Processed {len(image_files)} images in {total_time:.2f} seconds.")
    print(f"Total unique IDs assigned: {gallery.get_gallery_size()}")

    # 4. Group Images by ID
    if args.group_output:
        output_folder = args.output_folder
        print(f"Grouping images by ID into: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
        for pid, paths in image_id_mapping.items():
            pid_folder = os.path.join(output_folder, f"ID_{pid}")
            os.makedirs(pid_folder, exist_ok=True)
            for img_path in paths:
                img_name = os.path.basename(img_path)
                target_path = os.path.join(pid_folder, img_name)
                try:
                    # Copy file instead of re-saving to preserve original
                    shutil.copy2(img_path, target_path)
                except Exception as e:
                    print(f"Failed to copy {img_path} to {target_path}: {e}")
        print(f"Images grouped successfully in {output_folder}")

        # 5. Zip Output (Optional)
        if args.zip_output:
            zip_path_base = os.path.join(os.path.dirname(output_folder), args.zip_filename)
            print(f"Zipping output folder {output_folder} to {zip_path_base}.zip ...")
            try:
                shutil.make_archive(zip_path_base, 'zip', output_folder)
                print(f"✅ Folder zipped successfully to {zip_path_base}.zip")
            except Exception as e:
                print(f"Error zipping output folder: {e}")

    if not args.no_display:
         print("Closing plots...")
         plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder of person images for Re-Identification and grouping.")
    parser.add_argument('--input-folder', type=str, required=True, help='Path to the folder containing cropped person images.')
    parser.add_argument('--reid-config', type=str, default='configs/reid_config.yaml', help='Path to the ReID configuration file.')
    parser.add_argument('--use-re-ranking', action='store_true', default=True, help='Enable re-ranking for folder processing (default: True).')
    parser.add_argument('--no-re-ranking', action='store_false', dest='use_re_ranking', help='Disable re-ranking for folder processing.')
    parser.add_argument('--group-output', action='store_true', default=True, help='Group output images into folders by ID (default: True).')
    parser.add_argument('--no-group-output', action='store_false', dest='group_output', help='Do not group output images.')
    parser.add_argument('--output-folder', type=str, default='output/grouped_by_id', help='Folder to save grouped images.')
    parser.add_argument('--zip-output', action='store_true', help='Zip the output folder after grouping.')
    parser.add_argument('--zip-filename', type=str, default='grouped_images', help='Base name for the output zip file (without .zip).')
    parser.add_argument('--no-display', action='store_true', help='Do not display images during processing.')

    args = parser.parse_args()
    process_folder(args)