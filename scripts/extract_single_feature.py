# scripts/extract_single_feature.py
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Add project root to sys.path ---
# This allows importing modules from utils and models when running the script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# ------------------------------------

from utils.config_loader import load_app_config
from models.reid import ReIDModel # Import the ReID model class

def main(args):
    """Loads ReID model, extracts feature from a single image, and saves/prints it."""

    # --- 1. Load Configuration (Only need ReID part) ---
    try:
        # We load both configs but only use the 'reid' part
        config = load_app_config(reid_config_path=args.config)
        reid_cfg = config['reid']
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the config file exists at the specified path.")
        return
    except Exception as e:
        print(f"Error loading ReID configuration: {e}")
        return

    # --- 2. Initialize ReID Model ---
    try:
        print("Initializing ReID model...")
        reid_model = ReIDModel(reid_cfg)
        print("ReID model initialized successfully.")
    except (ImportError, FileNotFoundError, RuntimeError, Exception) as e:
        print(f"Error initializing ReID Model: {e}")
        print("Check model paths, checkpoints, and dependencies (including CLIP-ReID submodule).")
        return

    # --- 3. Check Input Image ---
    if not os.path.isfile(args.image):
        print(f"Error: Input image file not found at '{args.image}'")
        return

    # --- 4. Extract Feature ---
    print(f"Extracting feature from: {args.image}")
    # Use the single image extraction method which handles loading, transforms, and normalization
    feature_vector = reid_model.extract_feature_single(args.image)

    # --- 5. Process Output ---
    if feature_vector is None:
        print("Feature extraction failed. Please check logs for errors.")
        return

    if not isinstance(feature_vector, torch.Tensor) or feature_vector.ndim != 2 or feature_vector.shape[0] != 1:
         print(f"Error: Unexpected output format from feature extraction. Expected shape (1, D), got {feature_vector.shape if isinstance(feature_vector, torch.Tensor) else type(feature_vector)}")
         return

    feature_dim = feature_vector.shape[1]
    print(f"Successfully extracted feature vector!")
    print(f" - Shape: {feature_vector.shape}")
    # Verify L2 norm (should be close to 1.0)
    norm = torch.linalg.norm(feature_vector).item()
    print(f" - L2 Norm: {norm:.6f}")

    # Move tensor to CPU for saving/printing if it's on GPU
    feature_vector_cpu = feature_vector.cpu()

    # Print first few elements (optional)
    print(f" - First 5 elements: {feature_vector_cpu.numpy().flatten()[:5]}")


    # --- 6. Save Feature Vector (Optional) ---
    if args.output:
        output_path = args.output
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        try:
            if output_path.lower().endswith(".npy"):
                # Save as NumPy array
                np.save(output_path, feature_vector_cpu.numpy())
                print(f"Feature vector saved as NumPy array to: {output_path}")
            elif output_path.lower().endswith(".pt"):
                 # Save as PyTorch tensor
                 torch.save(feature_vector_cpu, output_path)
                 print(f"Feature vector saved as PyTorch tensor to: {output_path}")
            else:
                 # Default to PyTorch tensor if extension unknown/missing
                 if not output_path.lower().endswith(".pt"):
                      output_path += ".pt"
                 torch.save(feature_vector_cpu, output_path)
                 print(f"Output path had no standard extension, saved as PyTorch tensor to: {output_path}")

        except Exception as e:
            print(f"Error saving feature vector to {output_path}: {e}")

    # --- 7. Show Image (Optional) ---
    if args.show:
        try:
            img = Image.open(args.image)
            plt.figure(f"Input Image: {os.path.basename(args.image)}")
            plt.imshow(img)
            plt.title("Input Image")
            plt.axis('off')
            print("Displaying input image (close window to continue)...")
            plt.show() # Blocks until window is closed
        except FileNotFoundError:
            print(f"Error: Could not display image, file not found: {args.image}")
        except ImportError:
             print("Error: Cannot display image. Matplotlib or Pillow might be missing.")
        except Exception as e:
            print(f"Error displaying image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Re-Identification feature vector from a single image.")

    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to the input image file.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the extracted feature vector (e.g., output/feature.pt or output/feature.npy).')
    parser.add_argument('--config', type=str, default='configs/reid_config.yaml',
                        help='Path to the ReID configuration file.')
    parser.add_argument('--show', action='store_true',
                        help='Display the input image using matplotlib.')

    args = parser.parse_args()
    main(args)