# Person Detection, Tracking, and Re-Identification Pipeline

This project implements a pipeline for detecting people in a video, tracking them across frames, and assigning a unique Re-Identification (ReID) number to each person using YOLOv8 for detection/tracking and CLIP-ReID for feature extraction. It also includes functionality to process a folder of pre-cropped person images for ReID.

## Features

* **Detection & Tracking:** Uses YOLOv8 with ByteTrack for robust person detection and tracking.
* **Re-Identification:** Employs CLIP-ReID (ViT-B-16) to extract appearance features for person matching.
* **ID Assignment:** Assigns consistent ReID numbers to tracked individuals.
* **Multiple Input Modes:** Supports processing from video files, single images, folders of images, and **live video streams** (including YouTube Live via `streamlink`).
* **Optimized Pipeline:** Includes optional optimizations like configurable ReID batching, target FPS limiting, `torch.compile` (PyTorch 2.0+), and mixed-precision inference.
* **Re-ranking (Optional):** Integrates k-reciprocal re-ranking for potentially higher ReID accuracy (useful for image folder processing, computationally expensive for video/live).
* **Modular Structure:** Code is organized into modules (`configs`, `models`, `utils`, `scripts`) with a central runner script (`run.py`).

## Project Structure
```
person-reid-project/
│
├── CLIP-ReID/              # Git submodule (CLIP-ReID library)
├── configs/                # Configuration files (.yaml, .yml)
├── data/                   # Sample input data (optional)
├── models/                 # Model definitions & weights folder
│   ├── weights/            # Place .pt and .pth model files here
│   └── ...
├── utils/                  # Utility functions
├── scripts/                # Task-specific logic implementations
│   ├── process_video_full_pipeline.py
│   ├── extract_single_feature.py
│   ├── process_image_folder.py
│   └── process_live_stream.py # Script for live stream logic
├── run.py                  # <<< Main Runner Script >>>
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore              # Files/folders ignored by Git
```

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/zoe141004/dinosrep.git](https://github.com/zoe141004/dinosrep.git)
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

## Usage (`run.py`)

***Overall command***:
```bash
python run.py <task> [task-specific-arguments] [global-options]
```
*   **Tasks:** 
    + video: full pipeline processing video (detection + tracking + re-id)
    + folder: processing a folder of cropped images (extract vector and compare (with re-raking or not))
    + image: processing a single image (only extract vector from input image)
    + live: Process a live video stream. (Youtube)
*   **Global Options (usable with any task):**
```
--reid-config <path>: Specify a different ReID configuration file (default: configs/reid_config.yaml).
--yolo-config <path>: Specify a different YOLO/tracking configuration file (default: configs/yolo_config.yaml).
```
### 1. Processing a Video File (video)

Run the main pipeline script:

```bash
python run.py video -i <input_video_path> -o <output_video_path> [options]
```
Command-line arguments:
```
--i,--input: (Required) Path to the input video file.
--o,--output: Path to save the output video with tracking and ReID results (default: output/run_video_output.mp4).
--no-display: Add this flag to run without showing the video window.
--use-re-ranking / --no-re-ranking: Enable/disable re-ranking (overrides config, default uses config setting). Warning: Enabling significantly impacts performance.
--reid-batch-size <size>: Set ReID inference batch size (overrides config).
```
Example: 
```bash
python run.py video -i data/sample_video.mp4 -o output/result.mp4  --reid-batch-size 32
```
### 2. Processing a Folder of Cropped Images (folder)
Run the folder processing script:

```Bash
python run.py folder -if <input_folder_path> [options]
```
Command-line arguments:
```
-if, --input-folder: (Required) Path to the folder containing cropped images.
-of, --output-folder: Folder to save grouped images (default: output/run_grouped_output). Only used if grouping is enabled.
--use-re-ranking / --no-re-ranking: Enable/disable re-ranking (default: enabled for this task).
--group-output / --no-group-output: Enable/disable grouping output images by ID (default: enabled).
--zip-output: Create a zip archive of the output folder (requires grouping enabled).
--zip-filename <name>: Base name for the zip file (default: run_grouped_images).
--no-display: Disable displaying images during processing.
--reid-batch-size <size>: Set ReID inference batch size (overrides config).
```
Example:
```bash
python run.py folder -if data/my_cropped_images/ -of output/grouped_run/ --no-re-ranking --no-group-output
```
### 3. Extracting Feature from a Single Image (extract)

You can use the provided script to extract the ReID feature vector for a single person image.

Command:

```bash
python run.py extract -i <input_image_path> [options]
```
Arguments:
```
-i, --image: (Required) Path to the input image file.
-o, --output <output_feature_path.pt/.npy>: Save the extracted feature vector (optional). Use .pt for PyTorch tensor or .npy for NumPy array.
--show: Display the input image after processing.
```
Example
```bash
python run.py extract -i data/sample_images/person1.jpg -o output/person1_feature.pt --show
```
### 4. Process Live Stream (live)
Processes a live video stream from a direct URL (HLS, RTSP) or a YouTube Live page URL (requires streamlink installed).

Command
```bash
python run.py live -s <stream_url_or_youtube_url> [options]
```
Arguments:
```
-s, --stream-url: (Required) Direct stream URL (e.g., .m3u8, rtsp://...) OR a standard YouTube Live page URL. If a YouTube URL is provided, the script will attempt to use streamlink (must be installed) to find the direct stream URL.
-o, --output <output_video_path>: Save the processed stream to a video file (optional). Warning: This file can become very large for long-running streams.
--no-display: Disable the preview window.
--use-re-ranking / --no-re-ranking: Enable/disable re-ranking (overrides config, default: disabled for live task due to performance cost).
--reid-batch-size <size>: Set ReID inference batch size (overrides config).
```
Example:
```bash
streamlink --stream-url https://www.youtube.com/live/u4UZ4UvZXrg?si=IGUh4TDltIqq_xv7 best
```
-> *output:*
https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1745088095/ei/_5kDaIWTAsCp7OsPvafV4Qk/ip/2a09:bac1:7a80:50::247:f0/id/u4UZ4UvZXrg.1/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hls_chunk_host/rr3---sn-i3b7kn6s.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/playlist_duration/30/manifest_duration/30/bui/AccgBcPosS_OrxVcWeA1tslqWsgeefTn3wkiKtKxbYmkBoRrNGot7dF-liQoXnkRfv8DZF9s5GuwzlvJ/spc/_S3wKqnQc_j8s4-2PLM8nFm38U6_HfR1HiDtEKxv43gEcJU4nSujuq7cYrIYpW6IpZuf6UI/vprv/1/playlist_type/DVR/initcwndbps/1542500/met/1745066497,/mh/x5/mm/44/mn/sn-i3b7kn6s/ms/lva/mv/m/mvi/3/pl/64/rms/lva,lva/dover/11/pacing/0/keepalive/yes/fexp/51355912/mt/1745066182/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,xpc,playlist_duration,manifest_duration,bui,spc,vprv,playlist_type/sig/AJfQdSswRAIgG3tNJCdR9xRD4oRRZwKh3I2Jqwbzr__IGl1Z8acmNqoCIA8OJ7vQNzRuT8aOmGqoZYfRC9VHOBHYHS1hiWsvsE8f/lsparams/hls_chunk_host,initcwndbps,met,mh,mm,mn,ms,mv,mvi,pl,rms/lsig/ACuhMU0wRAIgbNBtLSeZUZVtu6cL0FbrB9t0ITAPqYw9T_x8KOIYdlACIFI1yBayW64soqeqJMjLuqn5kIN-RcaV65Enxgs-OE0G/playlist/index.m3u8
```bash
!python run.py live --stream-url [link above] --output output/youtube_live_output.mp4
```
*updated:* now you can use youtube live direct link:
```bash
!python run.py live --stream-url youtube.com/live/... --output output/youtube_live_output.mp4
```
**Configuration Details**
* configs/reid_config.yaml: Controls ReID model paths, input size, similarity thresholds, re-ranking parameters, processing interval (for video), and optimization flags.
* configs/yolo_config.yaml: Controls YOLO model path, detection/tracking confidence, IoU thresholds, tracker type, and device.
* configs/vit_clipreid_base.yml: Base configuration for CLIP-ReID model architecture (usually not modified unless changing fundamental model parameters).
**Notes**
* *Performance:* Processing speed depends heavily on GPU, input resolution, number of detections, and selected options (re-ranking, batch size). ReID is often the bottleneck.
* *Re-ranking:* Significantly increases accuracy for static images but drastically reduces FPS for video/live streams.
* *torch.compile:* Requires PyTorch 2.0+. May improve inference speed after an initial compilation warmup, but can sometimes cause issues (e.g., Triton errors) depending on the environment. Can be disabled in reid_config.yaml.
* *Memory:* Running YOLO and CLIP-ReID simultaneously requires significant GPU memory. Adjust reid_batch_size if you encounter Out-Of-Memory errors
