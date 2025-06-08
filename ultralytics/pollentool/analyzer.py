import os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO, RTDETR
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel # For type checking detail
# It's good practice to also import specific predictor if we need to reference its properties,
# but for now, direct model.predict should work.
# from ultralytics.models.rtdetr.predict import RTDETRPredictor
from . import utils
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1200000000


class PollenAnalyzer:
    """
    Analyzes pollen images using a YOLO or RT-DETR model to detect, count,
    and characterize pollen grains.
    """
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        model_arch: str = "yolo",
        conf_threshold: float = 0.25,
        microns_per_pixel: float = None,
        device: str = 'cpu',
        default_imgsz: int = 640 # NEW: Add a configurable default image size for tests/warmup
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_type_info = "Unknown"
        self.model_arch = model_arch.lower().strip()
        self.default_imgsz = default_imgsz # Store default image size
        print(f"**Attempting to load model using specified architecture: {self.model_arch}**")

        try:
            if self.model_arch == "rtdetr":
                self.model = RTDETR(model_path)
                self.model_type_info = "RT-DETR (Ultralytics)"
            elif self.model_arch == "yolo":
                self.model = YOLO(model_path)
                # ... (YOLO type info logic remains the same) ...
                if hasattr(self.model, 'model') and self.model.model is not None:
                    if isinstance(self.model.model, RTDETRDetectionModel):
                        self.model_type_info = "RT-DETR compatible (loaded via YOLO class)"
                    elif isinstance(self.model.model, DetectionModel):
                        self.model_type_info = "YOLO-based DetectionModel (Ultralytics)"
                        if hasattr(self.model.model, 'yaml') and 'yaml_file' in self.model.model.yaml:
                             self.model_type_info += f" - Config: {Path(self.model.model.yaml['yaml_file']).name}"
                        elif hasattr(self.model, 'ckpt_path') and self.model.ckpt_path:
                             self.model_type_info += f" - From: {Path(self.model.ckpt_path).name}"
                    else:
                        self.model_type_info = f"Ultralytics YOLO-compatible ({type(self.model.model).__name__})"
                else:
                    self.model_type_info = "YOLO (Ultralytics)"
            else:
                raise ValueError(f"Unsupported model_arch: '{self.model_arch}'. Please choose 'yolo' or 'rtdetr'.")

            self.model.to(device)
            print(f"**Model Architecture Confirmed**: {self.model_type_info}")

            # --- MODIFIED Dummy Prediction ---
            # Use a more appropriate image size for the dummy prediction,
            # especially crucial for RT-DETR.
            # self.default_imgsz (e.g., 640) should be used.
            print(f"Attempting dummy prediction with imgsz={self.default_imgsz} on device '{device}'...")
            dummy_input_np = np.zeros((self.default_imgsz, self.default_imgsz, 3), dtype=np.uint8)
            
            # For RT-DETR, the model itself or its predictor might handle warmup differently.
            # The `model.predict` call should trigger necessary internal warmups.
            self.model.predict(source=dummy_input_np, verbose=False, imgsz=self.default_imgsz)
            print(f"Model '{Path(model_path).name}' (specified arch: {self.model_arch}) loaded and tested successfully on device '{device}'.")

        except Exception as e:
            print(f"Error during model loading/initialization (specified arch: {self.model_arch}, device: '{device}'): {e}. Trying CPU fallback.")
            cpu_device = 'cpu'
            try:
                if self.model_arch == "rtdetr":
                    self.model = RTDETR(model_path)
                    self.model_type_info = "RT-DETR (Ultralytics, on CPU)"
                elif self.model_arch == "yolo":
                    self.model = YOLO(model_path)
                    current_yolo_type = "YOLO (Ultralytics, on CPU)"
                    if hasattr(self.model, 'model') and self.model.model is not None:
                        if isinstance(self.model.model, DetectionModel): current_yolo_type = "YOLO-based DetectionModel (Ultralytics, on CPU)"
                    self.model_type_info = current_yolo_type
                
                self.model.to(cpu_device)
                # --- MODIFIED Dummy Prediction for CPU Fallback ---
                print(f"Attempting dummy prediction with imgsz={self.default_imgsz} on CPU fallback...")
                dummy_input_np = np.zeros((self.default_imgsz, self.default_imgsz, 3), dtype=np.uint8)
                self.model.predict(source=dummy_input_np, verbose=False, imgsz=self.default_imgsz)
                print(f"Model '{Path(model_path).name}' (specified arch: {self.model_arch}) loaded and tested successfully on CPU fallback.")
                print(f"**Model Type (CPU Fallback)**: {self.model_type_info}")
            except Exception as e_cpu:
                # If the error persists even with a more reasonable imgsz,
                # it might indicate a deeper issue with 'bestdetr.pt' compatibility or corruption.
                print(f"Critical error: Model '{Path(model_path).name}' (specified arch: {self.model_arch}, imgsz: {self.default_imgsz}) could not be loaded even on CPU: {e_cpu}")
                raise

        # Rest of __init__
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_analysis_dir = self.output_dir / "plots_analysis"
        self.crops_dir = self.output_dir / "crops"
        self.temp_dir = self.output_dir / "temp_jpgs"

        self.conf_threshold = conf_threshold
        self.microns_per_pixel = microns_per_pixel

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory set to: {self.output_dir}")
        if self.microns_per_pixel:
            print(f"Microns per pixel set to: {self.microns_per_pixel} for actual size calculation.")
        else:
            print("Microns per pixel not provided. Actual size will not be calculated.")

    # ... (_process_single_image, _generate_summary_report, _save_detailed_report, process_input methods remain the same) ...
    # Ensure these methods are present from the previous version of the code.
    def _process_single_image(
        self,
        image_path: Path,
        save_plots: bool,
        save_crops: bool,
        all_detections_details: list,
        summary_data_accumulator: dict
    ):
        # This method's internal logic should largely remain the same,
        # assuming YOLO().predict() and RTDETR().predict() from Ultralytics
        # return results with a consistent structure for .boxes, .xyxy, .conf, .cls
        # and model.names is available.

        # print(f"Processing image: {image_path.name}...") # Less verbose
        base_image_stem = image_path.stem
        processed_image_path = image_path

        if image_path.suffix.lower() in [".tif", ".tiff"]:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            converted_path = utils.convert_tif_to_jpg(image_path, self.temp_dir)
            if converted_path:
                processed_image_path = converted_path
            else:
                print(f"Skipping TIF image due to conversion error: {image_path.name}")
                return 0

        try:
            # When calling predict on actual images, you might want to pass imgsz based on
            # how your model was trained, or let Ultralytics handle it.
            # For now, not passing imgsz here to use default behavior for actual images.
            results = self.model.predict(
                source=str(processed_image_path),
                conf=self.conf_threshold,
                verbose=False
            )
        except Exception as e:
            print(f"Error during model prediction for {processed_image_path.name} (Arch: {self.model_arch}): {e}")
            return 0

        if not results or len(results) == 0:
            return 0

        result = results[0]
        
        if save_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            plot_filename = self.plots_dir / f"{base_image_stem}_plot.jpg"
            try:
                plotted_image = result.plot()
                cv2.imwrite(str(plot_filename), plotted_image)
            except Exception as e:
                print(f"Error saving plot for {base_image_stem}: {e}")

        detections_on_image = 0
        crop_class_counters = defaultdict(int)

        if result.boxes is not None and len(result.boxes) > 0:
            for i, box_data in enumerate(result.boxes):
                if not (hasattr(box_data, 'xyxy') and hasattr(box_data, 'conf') and hasattr(box_data, 'cls')):
                    print(f"Warning: Unexpected box_data format in {image_path.name}. Skipping box {i}.")
                    continue
                if box_data.xyxy is None or box_data.conf is None or box_data.cls is None or \
                   len(box_data.xyxy) == 0 or len(box_data.conf) == 0 or len(box_data.cls) == 0:
                    # print(f"Warning: Incomplete/empty box_data in {image_path.name}. Skipping box {i}.") # Can be verbose
                    continue

                detections_on_image += 1
                xyxy = box_data.xyxy[0].cpu().numpy()
                conf = float(box_data.conf[0].cpu().numpy())
                cls_idx = int(box_data.cls[0].cpu().numpy())
                
                if self.model.names and cls_idx < len(self.model.names):
                    class_name = self.model.names[cls_idx]
                else:
                    print(f"Warning: model.names issue or cls_idx out of range ({cls_idx}). Using class index as name.")
                    class_name = f"class_{cls_idx}"

                pixel_radius = utils.calculate_pixel_radius_from_bbox(xyxy)
                actual_radius_microns = utils.calculate_actual_radius(pixel_radius, self.microns_per_pixel)

                detection_detail = {
                    "ImageName": image_path.name, "DetectionID": i, "ClassName": class_name,
                    "Confidence": round(conf, 4),
                    "BBOX_x1": int(xyxy[0]), "BBOX_y1": int(xyxy[1]),
                    "BBOX_x2": int(xyxy[2]), "BBOX_y2": int(xyxy[3]),
                    "PixelRadius (pixels)": round(pixel_radius, 2),
                }
                if actual_radius_microns is not None:
                    detection_detail["ActualRadius_microns (microns)"] = round(actual_radius_microns, 2)
                all_detections_details.append(detection_detail)

                if class_name not in summary_data_accumulator:
                    summary_data_accumulator[class_name] = {
                        "count": 0, "total_conf": 0.0,
                        "total_pixel_radius": 0.0, "total_actual_radius": 0.0,
                        "actual_radius_count": 0
                    }
                summary_data_accumulator[class_name]["count"] += 1
                summary_data_accumulator[class_name]["total_conf"] += conf
                summary_data_accumulator[class_name]["total_pixel_radius"] += pixel_radius
                if actual_radius_microns is not None:
                    summary_data_accumulator[class_name]["total_actual_radius"] += actual_radius_microns
                    summary_data_accumulator[class_name]["actual_radius_count"] += 1

                if save_crops:
                    if result.orig_img is not None:
                        detection_idx_for_class = crop_class_counters[class_name]
                        utils.save_crop_for_detection(
                            result.orig_img, xyxy, base_image_stem,
                            class_name, detection_idx_for_class, self.crops_dir
                        )
                        crop_class_counters[class_name] += 1
        return detections_on_image

    def _generate_summary_report(self, summary_data_accumulator: dict, total_detections_count: int) -> list:
        if total_detections_count == 0:
            # print("No detections found. Summary report will be empty.") # Less verbose
            utils.save_summary_csv([], self.output_dir / "summary_report.csv")
            return []
        summary_list = []
        for class_name, data in summary_data_accumulator.items():
            count = data["count"]
            avg_conf = data["total_conf"] / count if count > 0 else 0
            avg_pixel_radius = data["total_pixel_radius"] / count if count > 0 else 0
            percentage = (count / total_detections_count) * 100 if total_detections_count > 0 else 0
            row = {
                "PollenType": class_name, "Count": count,
                "Percentage (%)": round(percentage, 2),
                "AverageConfidence": round(avg_conf, 4),
                "AveragePixelRadius (pixels)": round(avg_pixel_radius, 2),
            }
            if self.microns_per_pixel and data["actual_radius_count"] > 0:
                avg_actual_radius = data["total_actual_radius"] / data["actual_radius_count"]
                row["AverageActualRadius_microns (microns)"] = round(avg_actual_radius, 2)
            summary_list.append(row)
        summary_list.sort(key=lambda x: x["Count"], reverse=True)
        utils.save_summary_csv(summary_list, self.output_dir / "summary_report.csv")
        return summary_list

    def _save_detailed_report(self, all_detections_details: list):
        if all_detections_details:
            utils.save_details_csv(all_detections_details, self.output_dir / "detection_details.csv")

    def process_input(
        self,
        input_path_str: str,
        save_plots: bool = False,
        save_detailed_csv: bool = False,
        save_crops: bool = False,
        generate_analysis_plots: bool = True
    ):
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            return

        all_detections_details = []
        summary_data_accumulator = {}
        total_detections_count = 0
        
        if save_plots: self.plots_dir.mkdir(parents=True, exist_ok=True)
        if save_crops: self.crops_dir.mkdir(parents=True, exist_ok=True)
        if generate_analysis_plots: self.plots_analysis_dir.mkdir(parents=True, exist_ok=True)

        image_files = []
        if input_path.is_file():
            if input_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                image_files.append(input_path)
            # else: print(f"Unsupported file type: {input_path.name}. Skipping.") # Less verbose
        elif input_path.is_dir():
            # print(f"Scanning directory: {input_path}") # Less verbose
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", ".tiff"):
                image_files.extend(list(input_path.glob(ext)))
        else:
            print(f"Error: Input path is not a file or directory: {input_path}")
            return

        if not image_files:
            print(f"No supported image files found in {input_path}.")
            return
        
        # print(f"Found {len(image_files)} images to process.") # Less verbose

        for image_file in image_files:
            detections_in_current_image = self._process_single_image(
                image_file, save_plots, save_crops,
                all_detections_details, summary_data_accumulator
            )
            total_detections_count += detections_in_current_image
        
        summary_list = self._generate_summary_report(summary_data_accumulator, total_detections_count)
        
        if save_detailed_csv:
            self._save_detailed_report(all_detections_details)

        if generate_analysis_plots and summary_list:
            # print("Generating analysis plots...") # Less verbose
            utils.plot_class_distribution_bar(
                summary_list, self.plots_analysis_dir / "class_distribution_bar.png"
            )
            utils.plot_class_distribution_pie(
                summary_list, self.plots_analysis_dir / "class_distribution_pie.png"
            )
            utils.plot_average_confidence_bar(
                summary_list, self.plots_analysis_dir / "average_confidence_bar.png"
            )
            # print(f"Analysis plots saved to: {self.plots_analysis_dir}") # Less verbose
        # elif generate_analysis_plots and not summary_list:
            # print("No summary data available to generate analysis plots.") # Less verbose

        print(f"**Pollen analysis processing complete for {self.model_arch} model.**")
        print(f"Total objects detected across all images: {total_detections_count}")

