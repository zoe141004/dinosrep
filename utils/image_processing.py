# utils/image_processing.py
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image

def build_reid_transforms(input_size_hw, pixel_mean, pixel_std):
    """Builds standard torchvision transforms for ReID (used mainly for single image processing)."""
    if not isinstance(input_size_hw, (list, tuple)) or len(input_size_hw) != 2:
         raise ValueError(f"input_size_hw must be a list or tuple of [height, width], got {input_size_hw}")
    normalize_transform = T.Normalize(mean=pixel_mean, std=pixel_std)
    transform = T.Compose([
        T.Resize(input_size_hw), # expects (h, w)
        T.ToTensor(), # Converts PIL image (H, W, C) [0, 255] to Tensor (C, H, W) [0.0, 1.0]
        normalize_transform
    ])
    return transform

def get_optimized_reid_transforms(pixel_mean, pixel_std):
    """Returns optimized normalization and ToTensor transforms for batch processing."""
    # ToTensor() converts HWC numpy/PIL (0-255) -> CHW tensor (0.0-1.0)
    to_tensor_transform = T.ToTensor()
    # Normalize operates on CHW tensors
    normalize_transform = T.Normalize(mean=pixel_mean, std=pixel_std)
    return normalize_transform, to_tensor_transform

def preprocess_batch_optimized(np_crops_rgb, target_size_wh, normalize_transform, to_tensor_transform, device):
    """
    Preprocesses a batch of NumPy RGB crops using optimized methods (cv2 resize).

    Args:
        np_crops_rgb (list): List of NumPy arrays (H, W, C) in RGB format.
        target_size_wh (tuple): Target size as (width, height) for cv2.resize.
        normalize_transform (callable): Torchvision normalize transform.
        to_tensor_transform (callable): Torchvision ToTensor transform.
        device (torch.device): Target device for the output tensor.

    Returns:
        torch.Tensor or None: Batch tensor (N, C, H, W) on the target device, or None if input is empty/invalid.
    """
    tensors = []
    target_w, target_h = target_size_wh # (width, height) for cv2.resize
    if not np_crops_rgb:
        return None

    for crop in np_crops_rgb:
        # Basic validation of the crop
        if crop is None or not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[0] < 1 or crop.shape[1] < 1:
            # print("Warning: Skipping invalid crop in preprocessing.") # Optional log
            continue # Skip this invalid crop

        try:
            # Resize using cv2 (often faster than PIL/torchvision on CPU for this task)
            # INTER_LINEAR is a good balance between speed and quality
            resized_crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Convert NumPy (H, W, C) RGB (0-255) to Tensor (C, H, W) (0.0-1.0)
            tensor = to_tensor_transform(resized_crop)

            # Normalize
            tensor = normalize_transform(tensor)
            tensors.append(tensor) # Collect individual tensors

        except Exception as e:
             print(f"Error during preprocessing single crop: {e}. Skipping crop.")
             continue


    if not tensors: # If all crops were invalid or failed
        return None

    # Stack tensors into a batch and move to target device
    # Using non_blocking=True might offer a small speedup for CPU->GPU transfers
    try:
         input_batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
         return input_batch
    except Exception as e:
         print(f"Error stacking tensors or moving to device: {e}")
         return None


def crop_image_numpy(image_numpy, bbox):
    """
    Crops a NumPy image (BGR or RGB) given a bounding box [x1, y1, x2, y2].
    Performs bounds checking.
    """
    if image_numpy is None or not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) < 4:
         return None

    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = image_numpy.shape[:2]

    # Clamp coordinates to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2) # Exclusive boundary for slicing width
    y2 = min(h, y2) # Exclusive boundary for slicing height

    # Check if the resulting box has valid dimensions
    if x1 >= x2 or y1 >= y2:
        # print(f"Warning: Invalid crop coordinates after clamping: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return None # Return None for invalid crop

    try:
         crop = image_numpy[y1:y2, x1:x2]
         # Double check if crop is empty after slicing (can happen with edge cases)
         if crop.shape[0] > 0 and crop.shape[1] > 0:
             return crop
         else:
             # print(f"Warning: Crop resulted in empty array for bbox {bbox} on image shape {h}x{w}")
             return None
    except Exception as e:
         print(f"Error during numpy cropping with bbox {bbox}: {e}")
         return None