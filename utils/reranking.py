# utils/reranking.py
import sys
import os
import time
import numpy as np
import torch

_reranking_available = False
clip_reid_reranking_func = None

# --- Try to import re_ranking function ---
# Determine the absolute path to the CLIP-ReID/utils directory
# This assumes utils/reranking.py is one level below the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
clip_reid_util_path = os.path.join(project_root, 'CLIP-ReID', 'utils')

if os.path.isdir(clip_reid_util_path):
    if clip_reid_util_path not in sys.path:
        print(f"Adding CLIP-ReID utils path to sys.path: {clip_reid_util_path}")
        sys.path.append(clip_reid_util_path)
    try:
        from reranking import re_ranking as clip_reid_reranking_func
        print("Successfully imported re_ranking function from CLIP-ReID submodule.")
        _reranking_available = True
    except ImportError as e:
        print(f"Warning: Could not import 're_ranking' from '{clip_reid_util_path}'. Re-ranking will be disabled. Error: {e}")
    except Exception as e:
        print(f"Warning: An unexpected error occurred during re_ranking import. Re-ranking will be disabled. Error: {e}")
else:
    print(f"Warning: CLIP-ReID utils directory not found at '{clip_reid_util_path}'. Did you initialize the submodule (`git submodule update --init --recursive`)? Re-ranking will be disabled.")
# --- End Import ---


def is_reranking_available():
    """Checks if the re_ranking function was successfully imported."""
    return _reranking_available

def perform_re_ranking(query_features, gallery_features, k1, k2, lambda_value):
    """
    Wrapper for the k-reciprocal re_ranking function from CLIP-ReID.
    Handles input/output types (assumes function expects NumPy arrays) and availability check.

    Args:
        query_features (torch.Tensor): Query features (N, D) on any device.
        gallery_features (torch.Tensor): Gallery features (M, D) on any device.
        k1 (int): Re-ranking parameter.
        k2 (int): Re-ranking parameter.
        lambda_value (float): Re-ranking parameter.

    Returns:
        torch.Tensor or None: Distance matrix (N, M) on the query features' original device,
                              or None if re-ranking is unavailable or fails. Lower values indicate higher similarity.
    """
    if not _reranking_available or clip_reid_reranking_func is None:
        # print("Re-ranking is not available.") # Avoid spamming logs
        return None

    if query_features is None or gallery_features is None or query_features.shape[0] == 0 or gallery_features.shape[0] == 0:
        # print("Warning: Empty query or gallery features provided for re-ranking.")
        return torch.empty((query_features.shape[0] if query_features is not None else 0,
                            gallery_features.shape[0] if gallery_features is not None else 0)) # Return empty tensor


    try:
        # print("Applying re-ranking...") # Avoid spamming logs
        start_time = time.time()

        # --- Convert tensors to NumPy arrays on CPU (Common requirement for cython/numpy based re-ranking) ---
        qf_device = query_features.device # Store original device
        qf_np = query_features.float().cpu().numpy() # Use float() for consistency
        gf_np = gallery_features.float().cpu().numpy()

        # Call the imported function
        dist_mat_np = clip_reid_reranking_func(qf_np, gf_np, k1, k2, lambda_value)

        # --- Convert distance matrix back to torch.Tensor on original device ---
        if isinstance(dist_mat_np, np.ndarray):
             dist_mat_tensor = torch.from_numpy(dist_mat_np).to(qf_device)
        else:
             # If the function unexpectedly returns a tensor, ensure it's on the right device
             dist_mat_tensor = dist_mat_np.to(qf_device) if hasattr(dist_mat_np, 'to') else None
             if dist_mat_tensor is None:
                  print("Warning: Re-ranking function returned unexpected type.")

        # print(f"Re-ranking completed in {time.time() - start_time:.4f} seconds.") # Avoid spamming logs
        return dist_mat_tensor

    except Exception as e:
        print(f"Error during re-ranking execution: {e}")
        # Consider logging the traceback for debugging
        # import traceback
        # traceback.print_exc()
        return None # Indicate failure