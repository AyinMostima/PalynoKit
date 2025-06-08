import os
import shutil  # 用于复制文件和删除目录
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import csv  # 确保导入csv

from ultralytics import YOLO, RTDETR
from ultralytics.nn.tasks import DetectionModel  # RTDETRDetectionModel (如果需要更精确的类型检查)
from ultralytics.engine.results import Results
from . import utils


class PollenPipelineAnalyzer:
    def __init__(
            self,
            detect_model_path: str,
            classify_model_path: str,
            output_dir: str,
            detect_model_arch: str = "yolo",
            detect_conf_threshold: float = 0.25,
            microns_per_pixel: float = None,
            device: str = 'cpu',
            default_imgsz: int = 640,
            detector_target_class_ids: list[int] | None = None
    ):
        self.output_dir = Path(output_dir)
        self.device = device
        self.default_imgsz = default_imgsz
        self.detect_conf_threshold = detect_conf_threshold
        self.microns_per_pixel = microns_per_pixel
        self.detector_target_class_ids = detector_target_class_ids

        # --- Load Detection Model ---
        self.detect_model_arch = detect_model_arch.lower().strip()
        self.detect_model_type_info = "Unknown Detector"
        print(f"**Attempting to load DETECTION model: {detect_model_path} (Arch: {self.detect_model_arch})**")
        try:
            if self.detect_model_arch == "rtdetr":
                self.detect_model = RTDETR(detect_model_path)
                self.detect_model_type_info = "RT-DETR (Ultralytics)"
            elif self.detect_model_arch == "yolo":
                self.detect_model = YOLO(detect_model_path)
                self.detect_model_type_info = "YOLO (Ultralytics)"
                if hasattr(self.detect_model, 'model') and isinstance(self.detect_model.model, DetectionModel):
                    self.detect_model_type_info = "YOLO-based DetectionModel (Ultralytics)"
            else:
                raise ValueError(
                    f"Unsupported detect_model_arch: '{self.detect_model_arch}'. Choose 'yolo' or 'rtdetr'.")

            self.detect_model.to(device)
            dummy_input_np = np.zeros((self.default_imgsz, self.default_imgsz, 3), dtype=np.uint8)
            self.detect_model.predict(source=dummy_input_np, verbose=False, imgsz=self.default_imgsz)
            print(f"DETECTION model loaded and tested successfully on '{device}'. Type: {self.detect_model_type_info}")
        except Exception as e:
            print(f"Error loading DETECTION model on '{device}': {e}. Trying CPU.")
            if not hasattr(self, 'detect_model') or self.detect_model is None:  # Ensure model object exists for .to()
                if self.detect_model_arch == "rtdetr":
                    self.detect_model = RTDETR(detect_model_path)
                elif self.detect_model_arch == "yolo":
                    self.detect_model = YOLO(detect_model_path)
            self.detect_model.to('cpu')
            dummy_input_np_cpu = np.zeros((self.default_imgsz, self.default_imgsz, 3), dtype=np.uint8)
            self.detect_model.predict(source=dummy_input_np_cpu, verbose=False, imgsz=self.default_imgsz)
            print(f"DETECTION model loaded and tested successfully on CPU. Type: {self.detect_model_type_info}")

        # --- Load Classification Model ---
        print(f"**Attempting to load CLASSIFICATION model: {classify_model_path}**")
        try:
            self.classify_model = YOLO(classify_model_path)  # Assuming YOLO class can load classification models
            self.classify_model.to(device)
            dummy_cls_input_np = np.zeros((224, 224, 3), dtype=np.uint8)
            self.classify_model.predict(source=dummy_cls_input_np, verbose=False)  # Warmup/test
            print(f"CLASSIFICATION model loaded and tested successfully on '{device}'.")
        except Exception as e:
            print(f"Error loading CLASSIFICATION model on '{device}': {e}. Trying CPU.")
            if not hasattr(self, 'classify_model') or self.classify_model is None:  # Ensure model object exists
                self.classify_model = YOLO(classify_model_path)
            self.classify_model.to('cpu')
            dummy_cls_input_np_cpu = np.zeros((224, 224, 3), dtype=np.uint8)
            self.classify_model.predict(source=dummy_cls_input_np_cpu, verbose=False)
            print(f"CLASSIFICATION model loaded and tested successfully on CPU.")

        # --- Setup Output Directories ---
        self.crops_detected_only_dir = self.output_dir / "crops_detected_only_Pollen"
        self.crops_classified_dir = self.output_dir / "crops_classified"
        self.detection_plots_dir = self.output_dir / "detection_plots_initial"
        self.plots_analysis_dir = self.output_dir / "plots_analysis_pipeline"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crops_detected_only_dir.mkdir(parents=True, exist_ok=True)
        self.crops_classified_dir.mkdir(parents=True, exist_ok=True)

        print(f"Pipeline output directory set to: {self.output_dir.resolve()}")

    def _preprocess_input_images(self, original_input_path: Path) -> Path | None:
        """
        Prepares a temporary directory with all input images converted to JPG.
        Returns the path to this temporary directory, or None on failure.
        """
        temp_pipeline_input_dir = self.output_dir / "temp_pipeline_input_jpgs"
        if temp_pipeline_input_dir.exists():  # Clean up from previous run if any
            shutil.rmtree(temp_pipeline_input_dir)
        temp_pipeline_input_dir.mkdir(parents=True, exist_ok=True)
        print(f"Preparing temporary input directory: {temp_pipeline_input_dir}")

        image_files_to_process = []
        if original_input_path.is_file():
            if original_input_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                image_files_to_process.append(original_input_path)
        elif original_input_path.is_dir():
            for ext_pattern in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
                image_files_to_process.extend(list(original_input_path.glob(ext_pattern)))

        if not image_files_to_process:
            print(f"No supported image files found in the original input: {original_input_path}")
            shutil.rmtree(temp_pipeline_input_dir)  # Clean up empty temp dir
            return None

        processed_count = 0
        for original_file_path in image_files_to_process:
            if original_file_path.suffix.lower() in [".tif", ".tiff"]:
                print(f"Converting TIF/TIFF: {original_file_path.name}")
                converted_jpg_path = utils.convert_tif_to_jpg(original_file_path, temp_pipeline_input_dir)
                if not converted_jpg_path:
                    print(f"Failed to convert {original_file_path.name}. Skipping.")
                    continue
                processed_count += 1
            else:  # JPG, JPEG, PNG
                try:
                    destination_path = temp_pipeline_input_dir / original_file_path.name
                    shutil.copy2(original_file_path, destination_path)
                    print(f"Copied non-TIF image: {original_file_path.name} to temp dir.")
                    processed_count += 1
                except Exception as e_copy:
                    print(f"Error copying {original_file_path.name} to temp dir: {e_copy}. Skipping.")
                    continue

        if processed_count == 0:
            print(f"No images were successfully preprocessed into {temp_pipeline_input_dir}.")
            shutil.rmtree(temp_pipeline_input_dir)
            return None

        print(f"Successfully preprocessed {processed_count} images into {temp_pipeline_input_dir}")
        return temp_pipeline_input_dir

    def _perform_detection_and_crop(self, image_path_in_temp_dir: Path, original_image_bgr: np.ndarray):
        """
        Performs detection on the image from the temporary JPG directory.
        `image_path_in_temp_dir` is the path of the JPG image being processed.
        """
        detected_crop_instances_info = []
        # base_image_stem should refer to the original image name before any conversion for consistency
        # We can derive this if we store original name, or just use the temp name if simpler.
        # For now, let's use the stem of the image being processed (which is a JPG in temp dir).
        base_image_stem = image_path_in_temp_dir.stem

        full_detection_results_obj = None

        try:
            detect_results_list = self.detect_model.predict(
                source=original_image_bgr,  # Use the loaded BGR data
                conf=self.detect_conf_threshold,
                verbose=False
            )
            if not detect_results_list:
                return None, detected_crop_instances_info
            full_detection_results_obj = detect_results_list[0]
        except Exception as e:
            print(f"Error during detection for {image_path_in_temp_dir.name}: {e}")
            return None, detected_crop_instances_info

        if not full_detection_results_obj.boxes:
            return full_detection_results_obj, detected_crop_instances_info

        detection_id_counter = 0
        for box_data in full_detection_results_obj.boxes:
            if not (hasattr(box_data, 'xyxy') and hasattr(box_data, 'conf') and hasattr(box_data, 'cls')): continue
            if not all(getattr(box_data, attr) is not None and len(getattr(box_data, attr)) > 0 for attr in
                       ['xyxy', 'conf', 'cls']): continue

            detected_class_id = int(box_data.cls[0].cpu().numpy())
            if self.detector_target_class_ids is not None and detected_class_id not in self.detector_target_class_ids:
                continue

            xyxy = box_data.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            h_img, w_img = original_image_bgr.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
            if x1 >= x2 or y1 >= y2: continue

            crop_img_bgr = original_image_bgr[y1:y2, x1:x2]
            if crop_img_bgr.size == 0: continue

            # Crop filename uses stem from the image in temp dir (which is original_stem.jpg)
            crop_filename = f"{base_image_stem}_detection_{detection_id_counter}.jpg"
            crop_save_path = self.crops_detected_only_dir / crop_filename
            try:
                cv2.imwrite(str(crop_save_path), crop_img_bgr)
            except Exception as e_write_crop:
                print(f"Error saving initial crop {crop_save_path}: {e_write_crop}")
                continue

            detected_crop_instances_info.append({
                "crop_path": crop_save_path,
                "original_bbox_xyxy": xyxy,
                "detection_confidence": float(box_data.conf[0].cpu().numpy()),
                "original_image_stem": base_image_stem,  # This is now the stem of the (possibly converted) JPG
                "detection_id": detection_id_counter
            })
            detection_id_counter += 1
        return full_detection_results_obj, detected_crop_instances_info

    def _perform_classification(self, crop_path: Path):
        try:
            cls_results = self.classify_model.predict(source=str(crop_path), verbose=False)
            if not cls_results or not hasattr(cls_results[0], 'probs') or cls_results[0].probs is None:
                print(f"Warning: Classification for {crop_path.name} yielded no 'probs' or it was None.")
                return None, None, None
            probs_obj = cls_results[0].probs
            if probs_obj.top1 is None or probs_obj.top1conf is None:
                print(f"Warning: Classification for {crop_path.name} 'probs' had None for top1/top1conf.")
                return None, None, None
            top1_idx = probs_obj.top1
            top1_conf = float(probs_obj.top1conf.cpu().numpy())
            classified_class_name = self.classify_model.names.get(top1_idx,
                                                                  f"cls_{top1_idx}") if self.classify_model.names else f"cls_{top1_idx}"
            return classified_class_name, top1_conf, probs_obj
        except Exception as e:
            print(f"Error during classification for {crop_path.name}: {e}")
            return None, None, None

    def process_input(
            self,
            input_path_str: str,
            save_final_classified_crops: bool = True,
            save_initial_detection_plots: bool = False,
            save_detailed_csv: bool = True,
            generate_analysis_plots: bool = True
    ):
        original_input_path = Path(input_path_str)
        if not original_input_path.exists():
            print(f"Error: Original input path does not exist: {original_input_path}")
            return

        # --- Stage 0: Preprocess input images (convert TIFs, copy others to a temp dir) ---
        temp_pipeline_input_dir = self._preprocess_input_images(original_input_path)
        if not temp_pipeline_input_dir:
            print("Pipeline processing cannot continue due to preprocessing errors.")
            return

        # Now, all images for the pipeline are JPGs in temp_pipeline_input_dir
        # List images from this temporary directory
        pipeline_image_files = list(temp_pipeline_input_dir.glob("*.jpg"))  # Glob for .jpg

        if not pipeline_image_files:
            print(f"No JPG images found in the temporary processing directory: {temp_pipeline_input_dir}")
            shutil.rmtree(temp_pipeline_input_dir)  # Clean up
            return
        print(f"Starting pipeline processing for {len(pipeline_image_files)} images from {temp_pipeline_input_dir}")

        all_pipeline_results = []
        summary_data_accumulator = {}
        total_classified_pollen_count = 0
        classified_crop_class_counters = defaultdict(int)

        if save_initial_detection_plots:
            self.detection_plots_dir.mkdir(parents=True, exist_ok=True)
        if generate_analysis_plots:
            self.plots_analysis_dir.mkdir(parents=True, exist_ok=True)

        for jpg_image_file_path in pipeline_image_files:  # Iterate over JPGs in temp dir
            print(f"Processing image: {jpg_image_file_path.name} (from temp dir)")

            img_bgr = cv2.imread(str(jpg_image_file_path))
            if img_bgr is None:
                print(f"Could not load image {jpg_image_file_path.name} from temp dir. Skipping.")
                continue

            # --- Stage 1: Detection and Cropping ---
            # Pass the path of the JPG image from the temp directory
            raw_detector_results, detected_instances_to_classify = self._perform_detection_and_crop(
                jpg_image_file_path, img_bgr
            )

            if save_initial_detection_plots and raw_detector_results is not None:
                try:
                    # Plot filename uses the stem of the JPG file (which is original_stem.jpg)
                    plot_filename = self.detection_plots_dir / f"{jpg_image_file_path.stem}_initial_detections.jpg"
                    plotted_detection_image = raw_detector_results.plot()
                    cv2.imwrite(str(plot_filename), plotted_detection_image)
                except Exception as e_plot:
                    print(f"Error saving initial detection plot for {jpg_image_file_path.name}: {e_plot}")

            if not detected_instances_to_classify:
                continue  # Move to the next image in the temp directory

            # --- Stage 2: Classification ---
            for instance_info in detected_instances_to_classify:
                crop_path = instance_info["crop_path"]
                classified_name, cls_conf, _ = self._perform_classification(crop_path)

                if classified_name is None:
                    continue

                total_classified_pollen_count += 1

                if save_final_classified_crops:
                    class_specific_crop_dir = self.crops_classified_dir / classified_name
                    class_specific_crop_dir.mkdir(parents=True, exist_ok=True)
                    current_class_count = classified_crop_class_counters[classified_name]
                    # Classified crop filename uses original_image_stem from instance_info
                    # (which is the stem of the JPG image processed)
                    classified_crop_filename = f"{instance_info['original_image_stem']}_{classified_name}_{current_class_count}.jpg"
                    final_crop_save_path = class_specific_crop_dir / classified_crop_filename
                    try:
                        # Re-read the crop from crops_detected_only_dir to save to classified dir
                        temp_crop_img = cv2.imread(str(crop_path))
                        if temp_crop_img is not None:
                            cv2.imwrite(str(final_crop_save_path), temp_crop_img)
                            classified_crop_class_counters[classified_name] += 1
                        else:
                            print(f"Warning: Could not read crop {crop_path} to save as classified.")
                    except Exception as e_save_final_crop:
                        print(f"Error saving final classified crop {final_crop_save_path}: {e_save_final_crop}")

                # Prepare data for reports
                original_bbox = instance_info["original_bbox_xyxy"]
                pixel_radius = utils.calculate_pixel_radius_from_bbox(original_bbox)
                actual_radius_microns = utils.calculate_actual_radius(pixel_radius, self.microns_per_pixel)
                detail_row = {
                    "ImageName": jpg_image_file_path.name,  # Report with the name of the processed JPG
                    "DetectionID_in_Image": instance_info["detection_id"],
                    "Original_BBOX_x1": int(original_bbox[0]), "Original_BBOX_y1": int(original_bbox[1]),
                    "Original_BBOX_x2": int(original_bbox[2]), "Original_BBOX_y2": int(original_bbox[3]),
                    "DetectionConfidence": round(instance_info["detection_confidence"], 4),
                    "ClassName_From_Classifier": classified_name,
                    "ClassificationConfidence": round(cls_conf, 4),
                    "PixelRadius (pixels)": round(pixel_radius, 2),
                }
                if actual_radius_microns is not None:
                    detail_row["ActualRadius_microns (microns)"] = round(actual_radius_microns, 2)
                all_pipeline_results.append(detail_row)

                # Accumulate summary data
                pollen_type_key = classified_name
                if pollen_type_key not in summary_data_accumulator:
                    summary_data_accumulator[pollen_type_key] = {
                        "count": 0, "total_cls_conf": 0.0,
                        "total_pixel_radius": 0.0, "total_actual_radius": 0.0,
                        "actual_radius_count": 0
                    }
                summary_data_accumulator[pollen_type_key]["count"] += 1
                summary_data_accumulator[pollen_type_key]["total_cls_conf"] += cls_conf
                summary_data_accumulator[pollen_type_key]["total_pixel_radius"] += pixel_radius
                if actual_radius_microns is not None:
                    summary_data_accumulator[pollen_type_key]["total_actual_radius"] += actual_radius_microns
                    summary_data_accumulator[pollen_type_key]["actual_radius_count"] += 1

        # --- End of loop for pipeline_image_files ---

        # --- Stage 3: Generate Reports ---
        if save_detailed_csv:
            self._save_pipeline_details_csv(all_pipeline_results)
        summary_list_for_plots = self._generate_pipeline_summary_report(summary_data_accumulator,
                                                                        total_classified_pollen_count)
        if generate_analysis_plots and summary_list_for_plots:
            print("Generating pipeline analysis plots...")
            utils.plot_class_distribution_bar(
                summary_list_for_plots, self.plots_analysis_dir / "pipeline_class_distribution_bar.png"
            )
            utils.plot_class_distribution_pie(
                summary_list_for_plots, self.plots_analysis_dir / "pipeline_class_distribution_pie.png"
            )
            utils.plot_average_confidence_bar(
                summary_list_for_plots, self.plots_analysis_dir / "pipeline_average_cls_confidence_bar.png"
            )
            print(f"Pipeline analysis plots saved to: {self.plots_analysis_dir.resolve()}")
        elif generate_analysis_plots and not summary_list_for_plots:
            print("No summary data to generate pipeline analysis plots.")

        # --- Stage 4: Clean up temporary input directory ---
        if temp_pipeline_input_dir and temp_pipeline_input_dir.exists():
            try:
                shutil.rmtree(temp_pipeline_input_dir)
                print(f"Successfully cleaned up temporary input directory: {temp_pipeline_input_dir}")
            except Exception as e_cleanup:
                print(f"Error cleaning up temporary input directory {temp_pipeline_input_dir}: {e_cleanup}")

        print(f"**Pollen pipeline processing complete.**")
        print(f"Total pollen instances successfully classified: {total_classified_pollen_count}")

    def _save_pipeline_details_csv(self, details_data: list):
        if not details_data:
            # print("No pipeline details to save.") # Can be verbose
            return
        headers = [
            "ImageName", "DetectionID_in_Image",
            "Original_BBOX_x1", "Original_BBOX_y1", "Original_BBOX_x2", "Original_BBOX_y2",
            "DetectionConfidence", "ClassName_From_Classifier", "ClassificationConfidence",
            "PixelRadius (pixels)"
        ]
        # Dynamically add 'ActualRadius_microns (microns)' if present in any row
        if any("ActualRadius_microns (microns)" in row for row in details_data):
            if "ActualRadius_microns (microns)" not in headers:
                headers.append("ActualRadius_microns (microns)")

        output_path = self.output_dir / "pipeline_detection_classification_details.csv"
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Use extrasaction='ignore' if rows might have more keys than headers (should not happen here)
                writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(details_data)
            print(f"Pipeline detailed report saved to: {output_path.resolve()}")
        except Exception as e:
            print(f"Error saving pipeline details CSV: {e}")

    def _generate_pipeline_summary_report(self, summary_data_accumulator: dict, total_classified_count: int) -> list:
        if total_classified_count == 0:
            print("No pollen classified. Pipeline summary report will be empty.")
            # Ensure utils.save_summary_csv handles empty list gracefully or create empty file
            utils.save_summary_csv([], self.output_dir / "pipeline_summary_report.csv")
            return []

        summary_list = []
        for class_name, data in summary_data_accumulator.items():
            count = data["count"]
            avg_cls_conf = data["total_cls_conf"] / count if count > 0 else 0.0
            avg_pixel_radius = data["total_pixel_radius"] / count if count > 0 else 0.0
            percentage = (count / total_classified_count) * 100 if total_classified_count > 0 else 0.0
            row = {
                "PollenType": class_name,
                "Count": count,
                "Percentage (%)": round(percentage, 2),
                "AverageConfidence": round(avg_cls_conf, 4),  # This is classification confidence
                "AveragePixelRadius (pixels)": round(avg_pixel_radius, 2),
            }
            if self.microns_per_pixel and data["actual_radius_count"] > 0:
                avg_actual_radius = data["total_actual_radius"] / data["actual_radius_count"]
                row["AverageActualRadius_microns (microns)"] = round(avg_actual_radius, 2)
            summary_list.append(row)

        summary_list.sort(key=lambda x: x["Count"], reverse=True)
        utils.save_summary_csv(summary_list, self.output_dir / "pipeline_summary_report.csv")
        return summary_list

