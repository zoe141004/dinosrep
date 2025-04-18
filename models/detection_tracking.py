# models/detection_tracking.py
import torch
import os
from ultralytics import YOLO

class DetectorTracker:
    def __init__(self, config):
        """
        Initializes the YOLO detector and tracker.

        Args:
            config (dict): The 'yolo' section of the application configuration.
        """
        self.config = config
        self.model_path = config['yolo_model_path']
        self.device = config['device'] # Device determined by config_loader

        if not os.path.exists(self.model_path):
             raise FileNotFoundError(f"YOLO model file not found: {self.model_path}")

        print(f"Loading YOLO model from: {self.model_path} onto device: {self.device}")
        try:
            self.model = YOLO(self.model_path)
            # No need to manually move model, YOLO handles device argument in track/predict
            # self.model.to(self.device)
            # Perform a dummy inference to check loading (optional but good practice)
            # self.model.predict(torch.zeros(1, 3, 640, 640).to(self.device), verbose=False)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

        # Store tracking parameters from config for use in track_frame
        self.tracker_type = config.get('tracker_type', 'bytetrack.yaml')
        self.persist = config.get('tracking_persist', True)
        self.imgsz = config.get('tracking_imgsz', 640)
        self.conf = config.get('tracking_conf', 0.4)
        self.iou = config.get('tracking_iou', 0.5)
        self.classes = config.get('tracking_classes', [0]) # Default to person class
        self.verbose = config.get('tracking_verbose', False)


    def track_frame(self, frame_rgb):
        """
        Performs detection and tracking on a single RGB frame.

        Args:
            frame_rgb (numpy.ndarray): Input frame in RGB format (H, W, C).

        Returns:
            ultralytics.engine.results.Results or None:
                The results object from model.track() containing boxes, IDs, etc.
                Returns None if tracking fails or no objects are detected/tracked.
        """
        if frame_rgb is None or frame_rgb.shape[0] == 0 or frame_rgb.shape[1] == 0:
             print("Error: Invalid input frame for tracking.")
             return None

        try:
            # Pass parameters directly to the track method
            results_list = self.model.track(
                source=frame_rgb,       # Input image
                imgsz=self.imgsz,       # Inference size
                conf=self.conf,         # Detection confidence threshold
                iou=self.iou,           # NMS IoU threshold
                classes=self.classes,   # Filter by class (e.g., 0 for person)
                device=self.device,     # Specify device for this inference run
                persist=self.persist,   # Keep track IDs between frames
                tracker=self.tracker_type,# Specify tracker configuration
                verbose=self.verbose    # Suppress excessive logging
            )
            # model.track returns a list of Results objects (usually just one for single image)
            return results_list[0] if results_list else None
        except Exception as e:
             print(f"Error during YOLO tracking execution: {e}")
             # import traceback
             # traceback.print_exc() # Uncomment for detailed debugging
             return None