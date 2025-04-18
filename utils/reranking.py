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
    Passes PyTorch tensors directly to the original function and handles availability check.

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
        # print("Re-ranking is not available.") # Optional log
        return None

    if query_features is None or gallery_features is None or query_features.shape[0] == 0 or gallery_features.shape[0] == 0:
        # print("Warning: Empty query or gallery features provided for re-ranking.")
        # Return empty tensor matching expected output dimensions if possible
        num_q = query_features.shape[0] if query_features is not None else 0
        num_g = gallery_features.shape[0] if gallery_features is not None else 0
        return torch.empty((num_q, num_g), device=query_features.device if query_features is not None else 'cpu')


    try:
        # print("Applying re-ranking...") # Optional log
        start_time = time.time()

        # --- QUAN TRỌNG: Truyền trực tiếp PyTorch tensors ---
        # Hàm re_ranking gốc sẽ xử lý tensor và chuyển sang numpy khi cần
        qf_device = query_features.device # Lưu lại device gốc
        # Đảm bảo dtype phù hợp nếu cần (thường float32)
        qf = query_features.float()
        gf = gallery_features.float().to(qf.device) # Đảm bảo cùng device

        # --- Xóa dòng print debug đã thêm trước đó ---
        # print(f"DEBUG: Type of clip_reid_reranking_func: {type(clip_reid_reranking_func)}")
        # print(f"DEBUG: Value of clip_reid_reranking_func: {clip_reid_reranking_func}")
        # ---------------------------------------------

        # Gọi hàm gốc với Pytorch Tensors
        dist_mat_np = clip_reid_reranking_func(qf, gf, k1, k2, lambda_value)

        # --- Convert kết quả (thường là NumPy array) về lại torch.Tensor trên device gốc ---
        if isinstance(dist_mat_np, np.ndarray):
             dist_mat_tensor = torch.from_numpy(dist_mat_np).to(qf_device)
        # Xử lý trường hợp hàm gốc trả về tensor (ít khả năng hơn với code bạn cung cấp)
        elif isinstance(dist_mat_np, torch.Tensor):
             dist_mat_tensor = dist_mat_np.to(qf_device)
        else:
             print(f"Warning: Re-ranking function returned unexpected type: {type(dist_mat_np)}")
             dist_mat_tensor = None # Hoặc xử lý lỗi phù hợp

        # print(f"Re-ranking completed in {time.time() - start_time:.4f} seconds.") # Optional log
        return dist_mat_tensor

    except TypeError as e: # Bắt lỗi cụ thể hơn nếu vẫn xảy ra
         print(f"TypeError during re-ranking execution: {e}")
         print("Double-check input types and the original re_ranking function.")
         import traceback
         traceback.print_exc()
         return None
    except Exception as e:
        print(f"Error during re-ranking execution: {e}")
        import traceback
        traceback.print_exc()
        return None # Indicate failure