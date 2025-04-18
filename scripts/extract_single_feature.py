# scripts/extract_single_feature.py
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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
    from models.reid import ReIDModel
except ImportError:
    # Handle import error only if running as main script
    if __name__ == '__main__':
        print("Error: Could not import project modules in standalone mode.")
        print("Ensure script is run from project root or PYTHONPATH is set.")
        sys.exit(1)
    else: # Allow import to fail if imported by run.py (it handles imports)
        pass


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! HÀM CHÍNH ĐỂ TRÍCH XUẤT FEATURE (ĐƯỢC GỌI BỞI run.py) !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def run_feature_extraction(reid_config, reid_model, args):
    """
    Extracts feature from a single image using a pre-initialized ReID model.

    Args:
        reid_config (dict): The 'reid' section of the configuration.
        reid_model (ReIDModel): Initialized ReID model instance.
        args (argparse.Namespace): Parsed arguments specific to the extract task.
    """
    print(f"Extracting feature from: {args.image}")
    if not os.path.isfile(args.image):
        print(f"Error: Input image file not found at '{args.image}'")
        return

    # --- Extract Feature using the pre-loaded model ---
    feature_vector = reid_model.extract_feature_single(args.image)

    # --- Process Output ---
    if feature_vector is None:
        print("Feature extraction failed. Check model logs or input image.")
        return

    if not isinstance(feature_vector, torch.Tensor) or feature_vector.ndim != 2 or feature_vector.shape[0] != 1:
         print(f"Error: Unexpected output format. Expected shape (1, D), got {feature_vector.shape if isinstance(feature_vector, torch.Tensor) else type(feature_vector)}")
         return

    feature_dim = feature_vector.shape[1]
    expected_dim = reid_config.get('expected_feature_dim', -1)
    if expected_dim != -1 and feature_dim != expected_dim:
         print(f"Warning: Extracted feature dimension ({feature_dim}) differs from expected ({expected_dim}). Check config.")

    print(f"Successfully extracted feature vector!")
    print(f" - Shape: {feature_vector.shape}")
    norm = torch.linalg.norm(feature_vector).item()
    print(f" - L2 Norm: {norm:.6f}") # Should be close to 1.0

    # Move tensor to CPU for saving/printing
    feature_vector_cpu = feature_vector.cpu()
    print(f" - First 5 elements: {feature_vector_cpu.numpy().flatten()[:5]}")

    # --- Save Feature Vector (Optional) ---
    if args.output:
        output_path = args.output
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                 print(f"Creating output directory: {output_dir}")
                 os.makedirs(output_dir)
            except OSError as e:
                 print(f"Error creating output directory {output_dir}: {e}")
                 # Continue without saving if directory creation fails

        if not output_dir or os.path.exists(output_dir): # Proceed only if dir exists or not needed
            try:
                file_ext = os.path.splitext(output_path)[1].lower()
                if file_ext == ".npy":
                    np.save(output_path, feature_vector_cpu.numpy())
                    print(f"Feature vector saved as NumPy array to: {output_path}")
                elif file_ext == ".pt":
                    torch.save(feature_vector_cpu, output_path)
                    print(f"Feature vector saved as PyTorch tensor to: {output_path}")
                else:
                    # Default to .pt if extension is missing or unknown
                    if file_ext == "": output_path += ".pt"
                    torch.save(feature_vector_cpu, output_path)
                    print(f"Output path had no standard extension, saved as PyTorch tensor to: {output_path}")
            except Exception as e:
                print(f"Error saving feature vector to {output_path}: {e}")

    # --- Show Image (Optional) ---
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

# ==============================================================
#     BLOCK ĐỂ CHẠY FILE NÀY ĐỘC LẬP (Standalone Mode)
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Re-Identification feature vector (Standalone Script)")
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to save the extracted feature vector (.pt or .npy).')
    parser.add_argument('--config', type=str, default='configs/reid_config.yaml', help='Path to the ReID configuration file.')
    parser.add_argument('--show', action='store_true', help='Display the input image.')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: Input image file not found at '{args.image}'")
    else:
        print("[Standalone Mode] Loading config and ReID model...")
        try:
            # Load config (only need reid part, but load_app_config loads both)
            config = load_app_config(reid_config_path=args.config)
            # Initialize ReID model
            reid_model = ReIDModel(config['reid'])
            print("[Standalone Mode] Model loaded.")
            # Call the main logic function
            run_feature_extraction(config['reid'], reid_model, args)
        except ImportError:
            # Error already handled by try-except block at top for standalone mode
            pass
        except (FileNotFoundError, RuntimeError, ValueError, Exception) as e:
            print(f"[Standalone Mode] Error during initialization or processing: {e}")
            import traceback
            traceback.print_exc()