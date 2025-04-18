Markdown

# Person Detection, Tracking, and Re-Identification Pipeline

This project implements a pipeline for detecting people in a video, tracking them across frames, and assigning a unique Re-Identification (ReID) number to each person using YOLOv8 for detection/tracking and CLIP-ReID for feature extraction. It also includes functionality to process a folder of pre-cropped person images for ReID.

## Features

* **Detection & Tracking:** Uses YOLOv11 with ByteTrack for robust person detection and tracking.
* **Re-Identification:** Employs CLIP-ReID (ViT-B-16) to extract appearance features.
* **ID Assignment:** Assigns consistent ReID numbers to tracked individuals across frames or across images in a folder.
* **Optimized Pipeline:** Includes optimizations like batch processing, `torch.compile` (PyTorch 2.0+), and optional mixed-precision inference for faster video processing.
* **Re-ranking (Optional):** Integrates k-reciprocal re-ranking for potentially higher ReID accuracy (especially useful for static image processing).
* **Modular Structure:** Code is organized into modules for configuration, models, utilities, and main scripts.

## Project Structure

person-reid-project/
│
├── CLIP-ReID/                # Git submodule (CLIP-ReID library)
├── configs/                  # Configuration files (.yaml, .yml)
├── data/                     # Sample input data (optional)
├── models/                   # Model definitions and weights folder
├── utils/                    # Utility functions (config, gallery, processing, etc.)
├── scripts/                  # Scripts for specific tasks (e.g., folder processing)
├── main_pipeline.py          # Main script for video processing
├── requirements.txt          # Python dependencies
└── README.md                 # This file


## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/zoe141004/dinosrep.git
    cd person-reid-project
    ```

2.  **Initialize Submodule:**
    The CLIP-ReID library is included as a Git submodule. Initialize it:
    ```bash
    git submodule update --init --recursive
    ```

3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate environment:
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    .\venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    Make sure you have the correct CUDA version installed if using GPU. Install the required Python packages:
    ```bash
    pip install --user -r requirements.txt
    ```
    *Note:* The `requirements.txt` specifies PyTorch versions. If you encounter issues, ensure the installed PyTorch version matches your CUDA version. You might need to manually install a specific PyTorch build from their website.

5.  **Download Model Weights:**
    * Download the CLIP-ReID checkpoint (e.g., `MSMT17_clipreid_12x12sie_ViT-B-16_60.pth`) and place it in the `models/weights/` directory.
    * Download the YOLOv8 model (e.g., `yolo11m.pt` or `yolov8n.pt`, etc.) and place it in the `models/weights/` directory.
    * *(If using DeepSORT instead of ByteTrack, download its weights as well).*

6.  **Verify Configuration:**
    * Check the paths in `configs/reid_config.yaml` and `configs/yolo_config.yaml`. Ensure `clip_reid_checkpoint` and `yolo_model_path` point to the correct files within `models/weights/`.
    * Adjust device settings (`'auto'`, `'cuda:0'`, `'cpu'`) in the config files if needed.

## Usage

### 1. Processing a Video File

Run the main pipeline script:

```bash
python main_pipeline.py --input path/to/your/video.mp4 --output output/processed_video.mp4
```
Command-line arguments:

--input: (Required) Path to the input video file.
--output: Path to save the output video with tracking and ReID results (default: output/result_video.mp4).
--reid-config: Path to the ReID configuration file (default: configs/reid_config.yaml).
--yolo-config: Path to the YOLO configuration file (default: configs/yolo_config.yaml).
--no-display: Add this flag to run without showing the video window.
2. Processing a Folder of Cropped Images
Run the folder processing script:

```Bash

python scripts/process_image_folder.py --input-folder path/to/your/cropped_images/ --output-folder output/grouped_images/
```
Command-line arguments:
```
--input-folder: (Required) Path to the folder containing pre-cropped person images.
--reid-config: Path to the ReID configuration file (default: configs/reid_config.yaml).
--use-re-ranking / --no-re-ranking: Enable or disable re-ranking (default: enabled for folder processing).
--group-output / --no-group-output: Enable or disable grouping output images by assigned ID (default: enabled).
--output-folder: Folder where grouped images will be saved (default: output/grouped_by_id).
--zip-output: Add this flag to create a zip archive of the output folder.
--zip-filename: Base name for the zip file (default: grouped_images).
--no-display: Add this flag to run without showing images using Matplotlib during processing.
```
Configuration Details
* configs/reid_config.yaml: Controls ReID model paths, input size, similarity thresholds, re-ranking parameters, processing interval (for video), and optimization flags.
* configs/yolo_config.yaml: Controls YOLO model path, detection/tracking confidence, IoU thresholds, tracker type, and device.
* configs/vit_clipreid_base.yml: Base configuration for CLIP-ReID model architecture (usually not modified unless changing fundamental model parameters).
**Notes**
* *Performance:* Video processing speed depends heavily on GPU capabilities, video resolution, and the number of people detected. ReID calculation is the most computationally intensive part. Adjusting reid_interval in reid_config.yaml affects the trade-off between FPS and ReID update frequency.
* *Re-ranking:* While potentially more accurate, re-ranking significantly increases computation time and is often disabled (use_re_ranking: False in reid_config.yaml) for real-time video processing. It's generally recommended for offline tasks like processing image folders.
* *torch.compile:* Requires PyTorch 2.0+. It can improve performance but might increase initial model loading time.
Memory: Running both YOLO and CLIP-ReID can consume significant GPU memory.

## Extracting Feature from a Single Image

You can use the provided script to extract the ReID feature vector for a single person image.

**Usage:**

```bash
python scripts/extract_single_feature.py -i path/to/your/person_image.jpg [options]
```
Arguments:
-i, --image: (Required) Path to the input image file.
-o, --output: (Optional) Path to save the extracted feature vector. Use .pt extension for PyTorch tensor format or .npy for NumPy format (e.g., output/image_feature.pt).
--config: (Optional) Path to the ReID configuration file (default: configs/reid_config.yaml).
--show: (Optional) Add this flag to display the input image after processing.
