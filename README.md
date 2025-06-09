# Pollen Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![framework](https://img.shields.io/badge/Framework-Ultralytics-brightgreen)](https://ultralytics.com/)

A tool for detecting and analyzing pollen grains in microscope images, built on the Ultralytics framework. It can process batches of images to detect, count, and measure pollen, generating detailed reports and visual outputs.

## Features

*   **Ultralytics Integration**: **Works with any model compatible with the Ultralytics framework**, including custom-trained models.
*   **Batch Processing**: Analyze entire folders of images with a single command.
*   **Detailed Output**: Generates annotated images, CSV reports with detection data, and individual cropped images of each detected object.
*   **Configurable**: Easily adjust parameters like confidence threshold, target device (CPU/GPU), and output types.

## Installation

Follow these steps to set up the environment. **Python 3.10 or newer is required**.

### 1. Clone the Repository
```bash
git clone https://github.com/AyinMostima/PalynoKit.git
cd PalynoKit
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment.

*   **Using `conda`:**
    ```bash
    conda create --name pollen-env python=3.10 -y
    conda activate pollen-env
    ```

*   **Using `venv`:**
    ```bash
    python3 -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

### 3. Install Dependencies
Install PyTorch according to your system's CUDA version for GPU support. Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for the correct command. Install this package and its remaining dependencies:

```bash
pip install -e .
pip install matplotlib seaborn # Required for generating analysis plots
```

### 4. (Optional) Install MMCV
For certain models (HieraEdgeNet-TADDH) or features that rely on the OpenMMLab ecosystem, you may need to install `mmcv`. This often requires a specific compiler version.

```bash
# Then, install openmim and mmcv
pip install -U openmim
mim install mmcv
```

## Quick Start Guide
### 1. Get a Model
Pre-trained models (e.g., HieraEdgeNet.pt, best.pt) are located in the https://huggingface.co/datasets/AyinMostima/HieraEdgeNetintegratesdatasets. Copy the desired model file to your project folder or note its path.
### 2. Prepare Data
Create a directory and place all the images you want to analyze inside it.
### 3. Run the Analysis
Create a new Python script (e.g., analyze.py) and use the following template.
```python
from pathlib import Path
from ultralytics.pollentool import PollenAnalyzer

# --- 1. Configuration ---
# Update these variables to match your setup.

# Display: **INPUT_FOLDER**: Path to the directory containing your source images.
INPUT_FOLDER = r"path/to/your/input/images"

# Display: **OUTPUT_FOLDER**: Path to the directory where results will be saved.
OUTPUT_FOLDER = r"path/to/your/output/results"

# Display: **MODEL_FILE**: Path to the .pt model file you want to use.
MODEL_FILE = r"path/to/your/HieraEdgeNet.pt"
MODEL_ARCH = "yolo"      # or "rtdetr"
# Confidence threshold for detections.
CONFIDENCE = 0.3

# Device to run the model on: 'cuda' for GPU, 'cpu' for CPU.
DEVICE = 'cuda'

# (Optional) Microns per pixel for size calculation. Set to None to disable.
MICRONS_PER_PIXEL = 0.25

# --- 2. Output Toggles ---
# Set to True or False to enable/disable specific outputs.

SAVE_PLOTS = True          # Save annotated images with bounding boxes.
SAVE_DETAILED_CSV = True   # Save a CSV with detailed data for each detection.
SAVE_CROPS = True          # Save cropped images of each detected pollen grain.
GENERATE_ANALYSIS_PLOTS = True # Generate summary plot charts (e.g., size distribution).


# --- 3. Run Analyzer ---
print("Starting analysis script...")
print(f"Input: {INPUT_FOLDER}")
print(f"Output: {OUTPUT_FOLDER}")
print(f"Model: {MODEL_FILE}")
print(f"**Model Architecture**: {MODEL_ARCH}")

try:
    # Initialize the analyzer
    analyzer = PollenAnalyzer(
        model_path=MODEL_FILE,
        output_dir=OUTPUT_FOLDER,
        conf_threshold=CONFIDENCE,
        microns_per_pixel=MICRONS_PER_PIXEL,
        device=DEVICE,
        model_arch=MODEL_ARCH,
    )

    # Process the input directory
    analyzer.process_input(
        input_path_str=INPUT_FOLDER,
        save_plots=SAVE_PLOTS,
        save_detailed_csv=SAVE_DETAILED_CSV,
        save_crops=SAVE_CROPS,
        generate_analysis_plots=GENERATE_ANALYSIS_PLOTS
    )
    print(f"Analysis complete. Results are in {Path(OUTPUT_FOLDER).resolve()}")

except FileNotFoundError as e:
    print(f"Error: {e}. Check if model path and input folder are correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```
### 4. Execute the Script
Run the script from your terminal (with the virtual environment activated):
```bash
python analyze.py
```


## Understanding the Output

After the script finishes, the specified OUTPUT_FOLDER will contain the following subdirectories:
  • /annotated_images/: Your original images with detection boxes drawn on them.
  • /crops/: Cropped images of every detected object, sorted into subfolders by class.
  • /detailed_results/: CSV files containing detailed information for every detection in each image.
  • analysis_plots.png: A summary image with charts if GENERATE_ANALYSIS_PLOTS is True.
  • summary_results.csv: A single CSV file summarizing counts from all processed images.

## Example Output Plots
When GENERATE_ANALYSIS_PLOTS is set to True, the script will generate several summary plots to help visualize the results across all processed images. Below are examples of these plots.
### Class Distribution (Pie Chart)
This plot provides a quick overview of the proportion of each detected pollen class, which is useful for understanding the overall composition.

<img src=".\example\pipeline_class_distribution_pie.png" alt="pipeline_class_distribution_pie" style="zoom:15%;" />





### Class Distribution (Bar Chart)
This bar chart displays the absolute count for each pollen class, sorted from the most to the least frequent, allowing for easy comparison of quantities.

<img src=".\example\pipeline_class_distribution_bar.png" alt="pipeline_class_distribution_bar" style="zoom:30%;" />

### Average Class Confidence
This chart shows the average confidence score for the detections within each class. It helps in assessing the model's performance and reliability on different pollen types.

<img src=".\example\pipeline_average_cls_confidence_bar.png" alt="pipeline_average_cls_confidence_bar" style="zoom:67%;" />



## Example Output Tables

Two main CSV files are produced as part of the detailed results:

### Detection Classification Details (pipeline_detection_classification_details.csv)

This CSV file records information for each detected pollen grain across all processed images. It includes bounding box coordinates, detection confidence, classification results, and size measurements.

| ImageName              | DetectionID_in_Image | Original_BBOX_x1 | Original_BBOX_y1 | Original_BBOX_x2 | Original_BBOX_y2 | DetectionConfidence | ClassName_From_Classifier | ClassificationConfidence | PixelRadius (pixels) | ActualRadius_microns (microns) |
| ---------------------- | -------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------------- | ------------------------- | ------------------------ | -------------------- | ------------------------------ |
| CM0402_01_tile1021.jpg | 0                    | 312              | 20               | 341              | 79               | 0.7791              | Platanus                  | 0.4003                   | 22                   | 5.5                            |
| CM0402_01_tile1042.jpg | 0                    | 19               | 0                | 70               | 24               | 0.7096              | Urticaceae_fam            | 0.955                    | 18.75                | 4.69                           |
| CM0402_01_tile1043.jpg | 1                    | 277              | 79               | 330              | 130              | 0.8088              | Urticaceae_fam            | 0.935                    | 26                   | 6.5                            |

### Summary Report (pipeline_summary_report.csv)

This table summarizes the counts and average statistics for each pollen type found across all images.

| PollenType     | Count | Percentage (%) | AverageConfidence | AveragePixelRadius (pixels) | AverageActualRadius_microns (microns) |
| -------------- | ----- | -------------- | ----------------- | --------------------------- | ------------------------------------- |
| Urticaceae_fam | 5839  | 24.64          | 0.7229            | 24.37                       | 6.09                                  |
| Taxus          | 3874  | 16.34          | 0.7260            | 19.10                       | 4.77                                  |
| Platanus       | 2253  | 9.51           | 0.6846            | 25.92                       | 6.48                                  |



## Ultralytics Framework Integration

A key feature of this tool is its direct compatibility with the Ultralytics ecosystem.
  • Use Any Ultralytics Model: You can use any model format supported by Ultralytics simply by providing the path to the .pt file in the MODEL_FILE variable.
  • Use Custom-Trained Models: If you train your own detection model using the Ultralytics training pipeline, the resulting best.pt from your training run can be used directly with this analysis script without any modification.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
