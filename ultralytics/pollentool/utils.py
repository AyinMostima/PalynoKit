import csv
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1200000000



# Ensure output messages are in English
def convert_tif_to_jpg(tif_path: Path, temp_dir: Path) -> Path | None:
    """
    Converts a TIF image to JPG format and saves it in a temporary directory.
    Returns the path to the converted JPG image or None on failure.
    """
    try:
        jpg_filename = tif_path.stem + ".jpg"
        jpg_path = temp_dir / jpg_filename
        temp_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(tif_path) as pil_img:
            # Handle common TIF modes, including 16-bit grayscale
            if pil_img.mode == 'I;16' or pil_img.mode == 'I;16B':
                # Convert to 8-bit grayscale then to RGB for wider compatibility
                pil_img = pil_img.convert('L').convert('RGB')
            elif pil_img.mode == 'CMYK':
                 pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'P': # Palette mode
                 pil_img = pil_img.convert('RGB')
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            cv_img = np.array(pil_img)
            bgr_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(jpg_path), bgr_img, [
                int(cv2.IMWRITE_JPEG_QUALITY), 100,
                int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ])
        print(f"Successfully converted: {tif_path} -> {jpg_path}")
        return jpg_path
    except Exception as e:
        print(f"Error converting {tif_path} to JPG: {e}")
        return None

def calculate_pixel_radius_from_bbox(bbox_xyxy: np.ndarray) -> float:
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return 0.0
    width = bbox_xyxy[2] - bbox_xyxy[0]
    height = bbox_xyxy[3] - bbox_xyxy[1]
    return (width + height) / 4.0

def calculate_actual_radius(pixel_radius: float, microns_per_pixel: float = None) -> float | None:
    if microns_per_pixel is not None and microns_per_pixel > 0:
        return pixel_radius * microns_per_pixel
    return None

def save_summary_csv(summary_data: list, output_path: Path):
    if not summary_data:
        print("No summary data to save.")
        return

    headers = [
        "PollenType", "Count", "Percentage (%)", "AverageConfidence",
        "AveragePixelRadius (pixels)"
    ]
    if summary_data and "AverageActualRadius_microns (microns)" in summary_data[0]: # Check for the new header key
         headers.append("AverageActualRadius_microns (microns)")

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row_data in summary_data:
                filtered_row = {header: row_data.get(header) for header in headers}
                writer.writerow(filtered_row)
        print(f"Summary report saved to: {output_path}")
    except Exception as e:
        print(f"Error saving summary CSV: {e}")

def save_details_csv(details_data: list, output_path: Path):
    if not details_data:
        print("No detailed data to save.")
        return

    headers = [
        "ImageName", "DetectionID", "ClassName", "Confidence",
        "BBOX_x1", "BBOX_y1", "BBOX_x2", "BBOX_y2", "PixelRadius (pixels)"
    ]
    if details_data and "ActualRadius_microns (microns)" in details_data[0]: # Check for new header key
        headers.append("ActualRadius_microns (microns)")
        
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row_data in details_data:
                filtered_row = {header: row_data.get(header) for header in headers}
                writer.writerow(filtered_row)
        print(f"Detailed report saved to: {output_path}")
    except Exception as e:
        print(f"Error saving details CSV: {e}")

def save_crop_for_detection(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    original_image_stem: str,
    class_name: str,
    detection_idx_for_class: int,
    crops_dir: Path
):
    try:
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        cropped_img = image_bgr[y1:y2, x1:x2]

        if cropped_img.size == 0:
            print(f"Warning: Crop for {class_name} in {original_image_stem} is empty. Skipping.")
            return

        crop_filename = f"{original_image_stem}_{class_name}_{detection_idx_for_class}.jpg"
        # Create class-specific subdirectory if it doesn't exist
        class_crop_dir = crops_dir / class_name
        class_crop_dir.mkdir(parents=True, exist_ok=True)
        crop_save_path = class_crop_dir / crop_filename
        
        cv2.imwrite(str(crop_save_path), cropped_img)
    except Exception as e:
        print(f"Error saving crop for {class_name} from {original_image_stem}: {e}")





# --- Plotting Functions ---
def _apply_nature_style(ax, title="", xlabel="", ylabel=""):
    """Helper to apply common styling elements."""
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=12) # General tick styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y') # Grid on y-axis often cleaner for bar
    # plt.tight_layout() # Call this after all plot elements are set

def plot_class_distribution_bar(summary_data: list, output_path: Path):
    if not summary_data:
        print("No data for class distribution bar plot.")
        return
    
    types = [item['PollenType'] for item in summary_data]
    counts = [item['Count'] for item in summary_data]

    if not types or not counts:
        print("Empty data for class distribution bar plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 0.5), 7)) # Adjusted height for rotated labels
    
    # MODIFIED: Address seaborn warning by using hue and legend=False
    sns.barplot(x=types, y=counts, ax=ax, hue=types, palette="viridis", dodge=False, legend=False) 
    
    _apply_nature_style(ax, title="Pollen Class Distribution", xlabel="Pollen Type", ylabel="Count")
    
    # MODIFIED: Rotate x-axis labels 90 degrees and set horizontal alignment
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    # ax.tick_params(axis='x', labelrotation=90) # Simpler way if ha default is fine
    
    plt.tight_layout() # Ensure everything fits

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution bar plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving class distribution bar plot: {e}")
    plt.close(fig)


def plot_class_distribution_pie(summary_data: list, output_path: Path):
    if not summary_data:
        print("No data for class distribution pie plot.")
        return

    # Ensure 'Count' exists and is numeric for sum
    total_counts = sum(item.get('Count', 0) for item in summary_data)
    
    # Handle cases where 'Percentage (%)' might be missing or needs recalculation
    percentages = []
    types = []
    for item in summary_data:
        count = item.get('Count', 0)
        if count > 0 : # Only include items with count > 0 in pie chart
            types.append(item['PollenType'])
            if 'Percentage (%)' in item:
                percentages.append(item['Percentage (%)'])
            elif total_counts > 0:
                percentages.append((count / total_counts) * 100)
            else:
                percentages.append(0) # Should not happen if count > 0

    if not types or not any(p > 0 for p in percentages):
        print("Empty or zero-value data for class distribution pie plot after filtering.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted for legend
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(types))) # Use a good colormap slice
    
    wedges, texts, autotexts = ax.pie(
        percentages, labels=None, autopct='%1.1f%%',
        startangle=140, counterclock=False, colors=colors,
        pctdistance=0.85, wedgeprops=dict(width=0.6, edgecolor='w') # Donut-like
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    ax.axis('equal')
    ax.set_title("Pollen Class Distribution (%)", fontsize=16, fontweight='bold', pad=20)
    
    # Improved legend handling
    ax.legend(wedges, types, title="Pollen Types", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10, title_fontsize='12')

    plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust for legend

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution pie chart saved to: {output_path}")
    except Exception as e:
        print(f"Error saving class distribution pie chart: {e}")
    plt.close(fig)


def plot_average_confidence_bar(summary_data: list, output_path: Path):
    if not summary_data:
        print("No data for average confidence bar plot.")
        return

    # Filter out entries where 'AverageConfidence' might be missing or not applicable
    filtered_data = [item for item in summary_data if 'AverageConfidence' in item and item.get('Count', 0) > 0]
    if not filtered_data:
        print("No valid data for average confidence bar plot after filtering.")
        return

    types = [item['PollenType'] for item in filtered_data]
    avg_confidences = [item['AverageConfidence'] for item in filtered_data]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 0.5), 7)) # Adjusted height
    
    # MODIFIED: Address seaborn warning
    sns.barplot(x=types, y=avg_confidences, ax=ax, hue=types, palette="mako", dodge=False, legend=False)
    
    _apply_nature_style(ax, title="Average Detection Confidence by Pollen Type",
                        xlabel="Pollen Type", ylabel="Average Confidence")
    ax.set_ylim(0, 1)
    
    # MODIFIED: Rotate x-axis labels 90 degrees and set horizontal alignment
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    
    plt.tight_layout() # Ensure everything fits

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Average confidence bar plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving average confidence bar plot: {e}")
    plt.close(fig)