# models/reid.py
import torch
import torch.nn.functional as F
import os
import sys
from PIL import Image
from torchvision import transforms # Keep torchvision for standard transforms
sys.path.append(r'D:\code\person-reid\CLIP-ReID')
# --- Import necessary components from CLIP-ReID submodule ---
# Ensure CLIP-ReID is importable (adjust path logic if project structure changes)
_clipreid_imported = False
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    clip_reid_root = os.path.join(project_root, 'CLIP-ReID')

    if os.path.isdir(clip_reid_root):
         if clip_reid_root not in sys.path:
             sys.path.append(clip_reid_root)
         # Import specific functions needed
         from model.make_model_clipreid import make_model
         print(f"CLIP-ReID path added: {clip_reid_root}")
         _clipreid_imported = True
    else:
         print(f"Error: CLIP-ReID directory not found at {clip_reid_root}. Submodule likely not initialized.")

except ImportError as e:
    print(f"Error importing from CLIP-ReID. Make sure submodule exists and dependencies are installed. Error: {e}")
    # Decide how to handle this - raise error or allow continuation without ReID?
    # raise ImportError("Could not import make_model from CLIP-ReID.") from e
except Exception as e:
     print(f"An unexpected error occurred during CLIP-ReID import: {e}")
     # raise

if not _clipreid_imported:
     # Define dummy function if import failed, allows rest of code to load but ReID will fail
     def make_model(*args, **kwargs):
          raise ImportError("CLIP-ReID components could not be imported.")
# --- End CLIP-ReID Import Handling ---


# Import utility functions from this project (using relative imports)
from utils.config_loader import load_clipreid_base_config
from utils.image_processing import build_reid_transforms, get_optimized_reid_transforms, preprocess_batch_optimized

class ReIDModel:
    def __init__(self, config):
        """
        Initializes the CLIP-ReID model based on the provided configuration.

        Args:
            config (dict): The 'reid' section of the application configuration.
        """
        if not _clipreid_imported:
             raise ImportError("Cannot initialize ReIDModel because CLIP-ReID components failed to import.")

        self.config = config
        self.device = config['device'] # Device from config_loader
        self.checkpoint_path = config['clip_reid_checkpoint']
        self.input_size_hw = tuple(config['reid_input_size']) # Expecting [H, W]

        if not os.path.exists(self.checkpoint_path):
             raise FileNotFoundError(f"ReID Checkpoint file not found: {self.checkpoint_path}")

        # --- Load and Configure CLIP-ReID Settings ---
        try:
             from config.defaults import _C as cfg # Import here again safely
             base_cfg = load_clipreid_base_config(config['clip_reid_config_base'])

             # Merge overrides from our app config into the CLIP-ReID cfg object
             base_cfg.defrost() # Allow modifications
             print(f"Initial config: {base_cfg.MODEL}")  # Debugging line
             if 'Transformer_TYPE' not in base_cfg.MODEL:
                print("Adding missing Transformer_TYPE to config.")
                base_cfg.MODEL.Transformer_TYPE = config.get('transformer_type', 'ViT-B-16')  # Default to 'ViT-B-16'
             base_cfg.MODEL.DEVICE = self.device
             base_cfg.MODEL.Transformer_TYPE = config.get('transformer_type', base_cfg.MODEL.Transformer_TYPE)
             base_cfg.MODEL.NAME = base_cfg.MODEL.Transformer_TYPE # Make sure NAME matches TYPE
             base_cfg.MODEL.SIE_CAMERA = config.get('sie_camera', base_cfg.MODEL.SIE_CAMERA)
             base_cfg.MODEL.SIE_VIEW = config.get('sie_view', base_cfg.MODEL.SIE_VIEW)
             base_cfg.MODEL.STRIDE_SIZE = config.get('stride_size', base_cfg.MODEL.STRIDE_SIZE)
             base_cfg.INPUT.SIZE_TEST = list(self.input_size_hw) # Ensure it's a list for cfg
             base_cfg.TEST.WEIGHT = self.checkpoint_path
             base_cfg.MODEL.PRETRAIN_PATH = self.checkpoint_path # Often needed by load_param
             # Add any other critical parameters from the original vit_clipreid.yml if missing
             # e.g., base_cfg.MODEL.ID_LOSS_TYPE = ... , base_cfg.MODEL.METRIC_LOSS_TYPE = ...
             # These might be needed by make_model depending on the CLIP-ReID version
             print(f"Updated config: {base_cfg.MODEL}")  # Debugging line
             base_cfg.freeze() # Freeze config after modifications
             self.clipreid_cfg = base_cfg # Store the final configured object
             print("CLIP-ReID configuration loaded and updated.")
        except ImportError:
             print("Error: Could not import CLIP_ReID.config.defaults._C")
             raise
        except Exception as e:
             print(f"Error configuring CLIP-ReID model settings: {e}")
             raise

        # --- Build and Load Model Weights ---
        self.num_classes = config.get('num_classes', 1041) # Number of classes from training dataset (e.g., MSMT17)
        print(f"Building CLIP-ReID model ({self.clipreid_cfg.MODEL.NAME}) for {self.num_classes} classes...")
        # camera_num and view_num might be needed by make_model
        camera_num = self.clipreid_cfg.MODEL.get('CAMERA_NUM', 15) # Get from cfg or default
        view_num = self.clipreid_cfg.MODEL.get('VIEW_NUM', 1)     # Get from cfg or default

        try:
             # Call the imported make_model function
             self.model = make_model(self.clipreid_cfg, num_class=self.num_classes, camera_num=camera_num, view_num=view_num)
        except Exception as e:
             print(f"Error calling make_model: {e}")
             print("Check if all necessary parameters are present in clipreid_cfg and num_classes is correct.")
             raise

        try:
            print(f"Loading ReID checkpoint: {self.clipreid_cfg.TEST.WEIGHT}")
            # Use the load_param method attached to the model instance
            self.model.load_param(self.clipreid_cfg.TEST.WEIGHT)
            print("ReID Checkpoint loaded successfully.")
        except FileNotFoundError:
             print(f"Error: Checkpoint file not found at {self.clipreid_cfg.TEST.WEIGHT}")
             raise
        except RuntimeError as e:
             print(f"Error loading checkpoint weights (likely mismatch): {e}")
             print("Ensure checkpoint matches model architecture (num_classes, layers, etc.)")
             raise
        except Exception as e:
             print(f"Unexpected error loading checkpoint weights: {e}")
             raise

        self.model.to(self.device)
        self.model.eval()
        print("ReID model moved to device and set to evaluation mode.")

        # --- Apply Torch Compile (Optional) ---
        self.use_torch_compile = config.get('use_torch_compile', False)
        if self.use_torch_compile:
            try:
                pt_version = torch.__version__.split('+')[0]
                major, minor = map(int, pt_version.split('.')[:2])
                if major >= 2:
                    print("Applying torch.compile to ReID model (mode='reduce-overhead')...")
                    # Choose mode: default, reduce-overhead, max-autotune
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("torch.compile applied successfully.")
                else:
                    print(f"torch.compile requires PyTorch 2.0+ (found {pt_version}). Skipping.")
                    self.use_torch_compile = False # Disable if not supported
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Continuing without compilation.")
                self.use_torch_compile = False # Disable on failure


        # --- Define Transforms ---
        # Store pixel mean/std for convenience
        self.pixel_mean = self.clipreid_cfg.INPUT.PIXEL_MEAN
        self.pixel_std = self.clipreid_cfg.INPUT.PIXEL_STD
        # Standard transforms (e.g., for single image processing)
        self.standard_transform = build_reid_transforms(self.input_size_hw, self.pixel_mean, self.pixel_std)
        # Optimized transforms (for batch processing with cv2 resize)
        self.optimized_normalize, self.optimized_totensor = get_optimized_reid_transforms(self.pixel_mean, self.pixel_std)

        # --- Mixed Precision Setting ---
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        if self.use_mixed_precision and 'cuda' not in str(self.device): # Check if device is CUDA
             print("Warning: Mixed precision (AMP) is only available on CUDA devices. Disabling.")
             self.use_mixed_precision = False
        if self.use_mixed_precision:
             print("Mixed precision inference enabled for ReID model.")


    @torch.no_grad()
    def extract_features_optimized(self, np_crops_rgb):
        """
        Extracts features from a batch of NumPy RGB crops using optimized preprocessing and inference.

        Args:
            np_crops_rgb (list): A list of NumPy arrays (H, W, C), RGB format.

        Returns:
            torch.Tensor or None: Batch of L2 normalized feature vectors (N, D) on the model's device,
                                  or None if input is invalid or processing fails.
        """
        if not np_crops_rgb:
            return None

        # --- 1. Preprocess Batch ---
        # Target size needs to be (Width, Height) for the cv2.resize used inside
        target_size_wh = (self.input_size_hw[1], self.input_size_hw[0])
        input_batch = preprocess_batch_optimized(
            np_crops_rgb, target_size_wh,
            self.optimized_normalize, self.optimized_totensor, self.device
        )

        if input_batch is None:
            # print("Optimized preprocessing failed or resulted in empty batch.") # Optional log
            return None

        # --- 2. Inference ---
        try:
            # Use autocast for mixed precision if enabled
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                # Model forward pass
                features = self.model(input_batch)

            # --- 3. L2 Normalize Features ---
            # Normalization is crucial for cosine similarity calculation
            features = F.normalize(features, p=2, dim=1)

            # Ensure output is float32 if mixed precision was used, some downstream tasks might expect it
            # if self.use_mixed_precision:
            #     features = features.float()

            return features # Features are already on self.device

        except Exception as e:
            print(f"Error during optimized ReID feature extraction inference: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for detailed debugging
            return None

    @torch.no_grad()
    def extract_feature_single(self, image_path):
         """
         Extracts L2 normalized feature for a single image file path using standard transforms.

         Args:
             image_path (str): Path to the image file.

         Returns:
             torch.Tensor or None: Feature vector (1, D) on the model's device, or None on error.
         """
         if not os.path.exists(image_path):
              print(f"Error: Image file not found at {image_path}")
              return None
         try:
             # Load image using PIL
             image = Image.open(image_path).convert('RGB')
             # Apply standard transforms (Resize, ToTensor, Normalize)
             input_tensor = self.standard_transform(image).unsqueeze(0).to(self.device)

             # Inference with optional mixed precision
             with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                  feature = self.model(input_tensor)

             # L2 Normalize
             feature = F.normalize(feature, p=2, dim=1)

             # if self.use_mixed_precision: feature = feature.float()

             return feature # Shape (1, D)

         except Exception as e:
              print(f"Error processing single image {image_path}: {e}")
              return None