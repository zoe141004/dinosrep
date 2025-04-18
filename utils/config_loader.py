# utils/config_loader.py
import yaml
import torch
import os

def load_app_config(reid_config_path='configs/reid_config.yaml', yolo_config_path='configs/yolo_config.yaml'):
    """Loads and merges ReID and YOLO configurations."""
    config = {}
    if not os.path.exists(reid_config_path):
        raise FileNotFoundError(f"ReID config file not found: {reid_config_path}")
    if not os.path.exists(yolo_config_path):
         raise FileNotFoundError(f"YOLO config file not found: {yolo_config_path}")

    try:
        with open(reid_config_path, 'r', encoding='utf-8') as f:
            config['reid'] = yaml.safe_load(f)
        with open(yolo_config_path, 'r', encoding='utf-8') as f:
            config['yolo'] = yaml.safe_load(f)
    except Exception as e:
        print(f"Error parsing config files: {e}")
        raise

    # Determine device automatically if set to 'auto'
    if config['reid'].get('device', 'auto') == 'auto':
        config['reid']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Add specific GPU index selection if needed, e.g., 'cuda:0', 'cuda:1'
        # You might want a more sophisticated device selection logic here based on available GPUs
    elif 'cuda' in config['reid']['device'] and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{config['reid']['device']}' requested but not available. Falling back to CPU.")
        config['reid']['device'] = 'cpu'


    if config['yolo'].get('device', 'auto') == 'auto':
        config['yolo']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Add specific GPU index selection if needed
    elif 'cuda' in config['yolo']['device'] and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{config['yolo']['device']}' requested but not available. Falling back to CPU.")
        config['yolo']['device'] = 'cpu'

    # Simple validation (add more as needed)
    if not os.path.exists(config['reid'].get('clip_reid_checkpoint')):
         raise FileNotFoundError(f"ReID checkpoint file not found at path specified in reid_config.yaml: {config['reid'].get('clip_reid_checkpoint')}")
    if not os.path.exists(config['yolo'].get('yolo_model_path')):
         raise FileNotFoundError(f"YOLO model file not found at path specified in yolo_config.yaml: {config['yolo'].get('yolo_model_path')}")
    if not os.path.exists(config['reid'].get('clip_reid_config_base')):
          raise FileNotFoundError(f"Base CLIP-ReID config file not found at path specified in reid_config.yaml: {config['reid'].get('clip_reid_config_base')}")

    print(f"Loaded ReID Config (Device: {config['reid']['device']})")
    print(f"Loaded YOLO Config (Device: {config['yolo']['device']})")
    return config

def load_clipreid_base_config(base_config_path):
    """Loads the base CLIP-ReID config from its YAML file."""
    # Import cfg late to avoid potential import loops if utils is imported early
    try:
         from config.defaults import _C as cfg # Import from submodule
    except ImportError:
         print("Error: Cannot import _C from CLIP_ReID.config.defaults. Make sure submodule is initialized.")
         raise
    except ModuleNotFoundError:
         print("Error: CLIP_ReID module not found. Make sure submodule exists and is in PYTHONPATH.")
         raise

    if not os.path.exists(base_config_path):
         raise FileNotFoundError(f"Error: Base CLIP-ReID config file not found at {base_config_path}")

    try:
        # Create a new config object to avoid modifying the global one if loaded multiple times
        temp_cfg = cfg.clone()
        temp_cfg.merge_from_file(base_config_path)
        print(f"Loaded base CLIP-ReID config from: {base_config_path}")
        return temp_cfg # Return the loaded cfg object
    except Exception as e:
        print(f"Error loading/merging base CLIP-ReID config: {e}")
        raise