# pollentool/two_stage_validator.py
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO, RTDETR
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from ultralytics.data.utils import check_det_dataset

class TwoStageValidator:
    def __init__(self,
                 dataset_yaml_path: str,
                 stage1_model_path: str,
                 stage2_cls_model_path: str,
                 stage1_model_arch: str = "yolo",
                 device: str = 'cpu',
                 imgsz: int = 640,
                 conf_thres_stage1: float = 0.25,
                 iou_thres_matching: float = 0.5
                 ):

        self.device = torch.device(device)
        self.imgsz = imgsz
        self.conf_thres_stage1 = conf_thres_stage1
        self.iou_thres_matching = iou_thres_matching

        print(f"Loading dataset from: {dataset_yaml_path}")
        try:
            dataset_info = check_det_dataset(dataset_yaml_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not load or parse dataset YAML '{dataset_yaml_path}'. Error: {e}")
            
        self.val_image_dir = Path(dataset_info['val'])
        dataset_root_path = Path(dataset_info.get('path', Path(dataset_yaml_path).parent))
        
        if not self.val_image_dir.is_absolute():
             self.val_image_dir = (dataset_root_path / self.val_image_dir).resolve()

        potential_label_dir_1 = dataset_root_path / 'labels' / self.val_image_dir.name
        potential_label_dir_2 = self.val_image_dir.parent.with_name('labels') / self.val_image_dir.name
        potential_label_dir_3 = Path(str(self.val_image_dir).replace("images", "labels"))

        if potential_label_dir_1.exists():
            self.val_label_dir = potential_label_dir_1
        elif potential_label_dir_2.exists():
            self.val_label_dir = potential_label_dir_2
        elif potential_label_dir_3.exists():
            self.val_label_dir = potential_label_dir_3
        else:
            self.val_label_dir = Path(str(self.val_image_dir).replace("images", "labels")) # Default

        if not self.val_image_dir.exists():
            raise FileNotFoundError(f"Validation image directory does not exist: {self.val_image_dir}")
        if not self.val_label_dir.exists():
            print(f"Attempted label directories:")
            print(f"  1: {potential_label_dir_1}")
            print(f"  2: {potential_label_dir_2}")
            print(f"  3: {potential_label_dir_3}")
            print(f"  Defaulted to: {self.val_label_dir}")
            raise FileNotFoundError(f"Validation label directory does not exist or could not be reliably inferred. Last attempt: {self.val_label_dir}")

        self.class_names = dataset_info['names'] 
        self.nc = int(dataset_info['nc'])
        print(f"Validation image directory: {self.val_image_dir}")
        print(f"Validation label directory: {self.val_label_dir}")
        print(f"Found {self.nc} classes: {self.class_names}")

        print(f"Loading Stage 1 ({stage1_model_arch}) detection model from: {stage1_model_path}")
        if stage1_model_arch.lower() == "rtdetr":
            self.stage1_model = RTDETR(stage1_model_path)
        elif stage1_model_arch.lower() == "yolo":
            self.stage1_model = YOLO(stage1_model_path)
        else:
            raise ValueError("stage1_model_arch must be 'yolo' or 'rtdetr'")
        self.stage1_model.to(self.device)
        self.stage1_model.eval()

        print(f"Loading Stage 2 classification model from: {stage2_cls_model_path}")
        self.stage2_model = YOLO(stage2_cls_model_path)
        self.stage2_model.to(self.device)
        self.stage2_model.eval()
        
        s2_model_names_map = self.stage2_model.names
        s2_model_names_list = list(s2_model_names_map.values()) if isinstance(s2_model_names_map, dict) else s2_model_names_map
        
        dataset_names_map = self.class_names
        dataset_names_list = list(dataset_names_map.values()) if isinstance(dataset_names_map, dict) else dataset_names_map

        if s2_model_names_list != dataset_names_list:
            print(f"Warning: Stage 2 model class names ({s2_model_names_list}) do not perfectly match dataset names ({dataset_names_list}). Ensure correct class mapping for metrics.")
            if len(s2_model_names_list) != len(dataset_names_list):
                raise ValueError(f"Stage 2 model ({len(s2_model_names_list)} classes) and dataset ({len(dataset_names_list)} classes) have different number of classes!")

        self.s2_cls_stats = {"correct": 0, "total": 0, 
                             "per_class_correct": np.zeros(self.nc, dtype=np.int32), 
                             "per_class_total": np.zeros(self.nc, dtype=np.int32)}

    def load_gt_labels(self, label_path: Path, img_shape):
        gt_boxes_norm = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    try:
                        values = list(map(float, line.strip().split()))
                        if len(values) == 5: 
                            cls_id = int(values[0])
                            if not (0 <= cls_id < self.nc):
                                print(f"Warning: Invalid class_id {cls_id} (nc={self.nc}) in {label_path} line {line_idx+1}. Skipping.")
                                continue
                            gt_boxes_norm.append(values)
                        elif len(values) == 4 and self.nc == 1:
                             gt_boxes_norm.append([0.0] + values)
                        else:
                            print(f"Warning: Malformed line in label file {label_path} line {line_idx+1}: {line.strip()} (expected 5 values or 4 for single class). Skipping.")
                    except ValueError:
                        print(f"Warning: Non-float value in label file {label_path} line {line_idx+1}: {line.strip()}. Skipping.")
        
        gt_classes_final = []
        gt_boxes_xyxy_final = []
        h, w = img_shape[:2]

        for gt_norm in gt_boxes_norm:
            cls_id, cx_norm, cy_norm, w_norm, h_norm = gt_norm
            
            x1 = (cx_norm - w_norm / 2) * w
            y1 = (cy_norm - h_norm / 2) * h
            x2 = (cx_norm + w_norm / 2) * w
            y2 = (cy_norm + h_norm / 2) * h
            gt_boxes_xyxy_final.append([x1, y1, x2, y2])
            gt_classes_final.append(int(cls_id))
            
        return torch.tensor(gt_boxes_xyxy_final, device=self.device, dtype=torch.float32) if gt_boxes_xyxy_final else torch.empty((0,4), device=self.device, dtype=torch.float32), \
               torch.tensor(gt_classes_final, device=self.device, dtype=torch.long) if gt_classes_final else torch.empty(0, device=self.device, dtype=torch.long)

    def _preprocess_image(self, img_path):
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            print(f"Warning: Could not read image {img_path}")
        return img0 

    def _get_stage2_classification(self, crop_bgr):
        if crop_bgr is None or crop_bgr.shape[0] == 0 or crop_bgr.shape[1] == 0:
            return None, 0.0
        try:
            results = self.stage2_model.predict(source=crop_bgr, device=self.device, verbose=False)
            if results and results[0].probs is not None:
                pred_cls_id = results[0].probs.top1 
                pred_conf = results[0].probs.top1conf.item()
                return pred_cls_id, pred_conf
        except Exception:
            pass
        return None, 0.0

    @torch.no_grad()
    def validate(self):
        image_files = sorted([p for p in self.val_image_dir.glob('*.*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']])
        if not image_files:
            print(f"No supported images found in {self.val_image_dir}")
            return None, None 

        confusion_matrix = ConfusionMatrix(nc=self.nc)
        pbar = tqdm(image_files, desc="Validating images")

        for img_idx, img_path in enumerate(pbar):
            label_path = self.val_label_dir / (img_path.stem + '.txt')
            img0 = self._preprocess_image(img_path)
            if img0 is None: 
                pbar.set_postfix_str(f"Skipped {img_path.name} (read error)")
                continue
            
            h0, w0 = img0.shape[:2]
            # gt_boxes_abs: tensor [M, 4] (xyxy)
            # gt_classes_abs: tensor [M] (cls_indices)
            gt_boxes_abs, gt_classes_abs = self.load_gt_labels(label_path, img0.shape) 
            
            try:
                s1_results = self.stage1_model.predict(source=str(img_path), device=self.device, conf=self.conf_thres_stage1, imgsz=self.imgsz, verbose=False)
            except Exception as e:
                print(f"Error in Stage 1 prediction for {img_path.name}: {e}")
                continue
            
            current_image_combined_preds_boxes_list = []
            current_image_combined_preds_scores_list = []
            current_image_combined_preds_classes_list = []

            if s1_results and s1_results[0].boxes is not None and len(s1_results[0].boxes) > 0:
                s1_boxes_data = s1_results[0].boxes.data
                
                for s1_box_row_idx, s1_box_row in enumerate(s1_boxes_data):
                    s1_xyxy = s1_box_row[:4]
                    s1_conf = s1_box_row[4].item()

                    x1, y1, x2, y2 = s1_xyxy.tolist()
                    x1_clamped, y1_clamped = max(0, int(x1)), max(0, int(y1))
                    x2_clamped, y2_clamped = min(w0, int(x2)), min(h0, int(y2))

                    if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
                        continue
                    crop = img0[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

                    s2_pred_cls_id, s2_pred_conf = self._get_stage2_classification(crop)

                    if s2_pred_cls_id is not None and 0 <= s2_pred_cls_id < self.nc:
                        combined_conf = s1_conf * s2_pred_conf
                        
                        current_image_combined_preds_boxes_list.append(s1_xyxy.cpu().numpy())
                        current_image_combined_preds_scores_list.append(combined_conf)
                        current_image_combined_preds_classes_list.append(s2_pred_cls_id)

                        if gt_boxes_abs.numel() > 0:
                            try:
                                ious_with_gt = box_iou(s1_xyxy.unsqueeze(0), gt_boxes_abs)[0]
                                if ious_with_gt.numel() > 0:
                                    best_iou_s1_gt, best_gt_idx = ious_with_gt.max(0)
                                    if best_iou_s1_gt.item() > self.iou_thres_matching:
                                        true_class_of_matched_gt = gt_classes_abs[best_gt_idx].item()
                                        if 0 <= true_class_of_matched_gt < self.nc:
                                            self.s2_cls_stats["total"] += 1
                                            self.s2_cls_stats["per_class_total"][true_class_of_matched_gt] += 1
                                            if s2_pred_cls_id == true_class_of_matched_gt:
                                                self.s2_cls_stats["correct"] += 1
                                                self.s2_cls_stats["per_class_correct"][true_class_of_matched_gt] +=1
                            except Exception:
                                pass 
            
            # detections_tensor: combined predictions for the current image [N, 6] (xyxy, conf, cls_idx)
            detections_tensor = torch.empty((0,6), device=self.device)
            if current_image_combined_preds_classes_list:
                pred_boxes_np = np.array(current_image_combined_preds_boxes_list)
                pred_scores_np = np.array(current_image_combined_preds_scores_list)
                pred_classes_np = np.array(current_image_combined_preds_classes_list)
                
                detections_tensor = torch.cat([
                    torch.from_numpy(pred_boxes_np).float(),
                    torch.from_numpy(pred_scores_np).float().unsqueeze(1),
                    torch.from_numpy(pred_classes_np).float().unsqueeze(1) # CM expects float classes for pred_cls
                ], dim=1).to(self.device)

            # For ConfusionMatrix.process_batch, gt_cls needs to be separate from gt_bboxes.
            # gt_boxes_abs is [M, 4] (xyxy)
            # gt_classes_abs is [M] (cls_indices)
            
            # ***** CORRECTED CALL HERE *****
            # process_batch(self, detections, gt_bboxes, gt_cls)
            if detections_tensor.numel() > 0 or gt_boxes_abs.numel() > 0: # Only process if there's something to process
                 confusion_matrix.process_batch(detections_tensor, gt_boxes_abs, gt_classes_abs.float()) # gt_cls as float for CM

        print("\n--- Two-Stage Validation Results ---")
        
        if self.s2_cls_stats["total"] > 0:
            overall_s2_acc = self.s2_cls_stats["correct"] / self.s2_cls_stats["total"] * 100
            print(f"Stage 2 Classifier Accuracy (on S1 IoU-matched crops): {overall_s2_acc:.2f}% ({self.s2_cls_stats['correct']}/{self.s2_cls_stats['total']})")
            for i in range(self.nc):
                class_name_str = self.class_names.get(i, f"Class_{i}")
                if self.s2_cls_stats["per_class_total"][i] > 0:
                    acc_cls = self.s2_cls_stats["per_class_correct"][i] / self.s2_cls_stats["per_class_total"][i] * 100
                    print(f"  - S2 Acc for class '{class_name_str}': {acc_cls:.2f}% ({int(self.s2_cls_stats['per_class_correct'][i])}/{int(self.s2_cls_stats['per_class_total'][i])})")
        else:
            print("Stage 2 Classifier Accuracy: No S1-matched GT crops were evaluated.")
        
        raw_ap_stats = [s for s in confusion_matrix.stats if s[0] is not None]

        map_overall, map50, map75 = 0.0, 0.0, 0.0 
        p_mean, r_mean = 0.0, 0.0
        ap50_per_class = {self.class_names.get(i, f"Class_{i}"): 0.0 for i in range(self.nc)}
        ap75_per_class = {self.class_names.get(i, f"Class_{i}"): 0.0 for i in range(self.nc)}

        if len(raw_ap_stats) > 0:
            print(f"\nCalculating mAP metrics (using {len(raw_ap_stats)} stats entries from ConfusionMatrix)...")
            try:
                save_dir_ap_plots = Path(self.val_image_dir.parent.parent, 'ap_plots_two_stage')
                save_dir_ap_plots.mkdir(parents=True, exist_ok=True)
                
                p, r, ap, f1, ap_class_all_ious = ap_per_class(
                    *zip(*raw_ap_stats),
                    plot=True, 
                    save_dir=save_dir_ap_plots,
                    names=self.class_names,
                    nc=self.nc 
                )
                
                p_mean, r_mean = p.mean().item(), r.mean().item()
                ap50_tensor = ap_class_all_ious[:, 0]
                ap75_idx = min(5, ap_class_all_ious.shape[1] - 1) 
                ap75_tensor = ap_class_all_ious[:, ap75_idx]
                
                map_overall = ap.mean().item()
                map50 = ap50_tensor.mean().item()
                map75 = ap75_tensor.mean().item()

                print(f"\n--- End-to-End Detection Metrics (S1 Box + S2 Class) ---")
                print(f"Total Processed Detections for mAP: {len(raw_ap_stats)}")
                print(f"Precision (mean): {p_mean:.4f}")
                print(f"Recall (mean):    {r_mean:.4f}")
                print(f"mAP@0.5-0.95: {map_overall:.4f}")
                print(f"mAP@0.50:     {map50:.4f}")
                print(f"mAP@0.75:     {map75:.4f}")
                
                print("\nPer-class AP@0.50:")
                for i in range(self.nc):
                    class_name_str = self.class_names.get(i, f"Class_{i}")
                    ap50_val = ap50_tensor[i].item()
                    ap50_per_class[class_name_str] = ap50_val
                    print(f"  - {class_name_str}: {ap50_val:.4f}")

                print("\nPer-class AP@0.75:")
                for i in range(self.nc):
                    class_name_str = self.class_names.get(i, f"Class_{i}")
                    ap75_val = ap75_tensor[i].item()
                    ap75_per_class[class_name_str] = ap75_val
                    print(f"  - {class_name_str}: {ap75_val:.4f}")

            except Exception as e_map:
                print(f"Error during mAP calculation with ap_per_class: {e_map}")
                import traceback
                traceback.print_exc()
        else:
            print("Not enough stats collected by ConfusionMatrix for mAP calculation (raw_ap_stats is empty).")
        
        metrics_summary = {
            "s2_accuracy_overall": (self.s2_cls_stats["correct"] / self.s2_cls_stats["total"] * 100) if self.s2_cls_stats["total"] > 0 else 0,
            # "s2_cls_stats_raw": self.s2_cls_stats, # Full stats dict if needed
            "mAP_0.5_0.95": map_overall,
            "mAP_0.50": map50,
            "mAP_0.75": map75,
            "mean_precision": p_mean,
            "mean_recall": r_mean,
            "per_class_ap50": ap50_per_class,
            "per_class_ap75": ap75_per_class,
        }
        s2_per_class_acc_dict = {}
        for i in range(self.nc):
            class_name_str = self.class_names.get(i, f"Class_{i}")
            if self.s2_cls_stats["per_class_total"][i] > 0:
                acc_cls = self.s2_cls_stats["per_class_correct"][i] / self.s2_cls_stats["per_class_total"][i] * 100
                s2_per_class_acc_dict[class_name_str] = acc_cls
            else:
                s2_per_class_acc_dict[class_name_str] = 0.0 # Or None or "N/A"
        metrics_summary["s2_accuracy_per_class"] = s2_per_class_acc_dict

        return metrics_summary


# --- Example Usage ---
if __name__ == "__main__":
    dummy_yaml_content = """
path: ../tests/data_twostage 
train: images/train 
val: images/val
nc: 2
names: {0: 'classA', 1: 'classB'}
    """
    dummy_yaml_path = Path("./dummy_pollen_dataset_twostage.yaml")
    with open(dummy_yaml_path, "w") as f:
        f.write(dummy_yaml_content)
    
    dataset_root = Path("../tests/data_twostage")
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    if not list((dataset_root / "images" / "val").glob("*.jpg")):
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.rectangle(dummy_img, (50,50), (100,100), (0,255,0), -1) 
        cv2.rectangle(dummy_img, (150,150), (200,200), (0,0,255), -1) 
        cv2.imwrite(str(dataset_root / "images" / "val" / "dummy_test_img_01.jpg"), dummy_img)
        with open(dataset_root / "labels" / "val" / "dummy_test_img_01.txt", "w") as f:
            f.write(f"0 {75/320:.4f} {75/320:.4f} {50/320:.4f} {50/320:.4f}\n")
            f.write(f"1 {175/320:.4f} {175/320:.4f} {50/320:.4f} {50/320:.4f}\n")
        
        dummy_img2 = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.rectangle(dummy_img2, (30,30), (80,80), (0,255,0), -1) 
        cv2.imwrite(str(dataset_root / "images" / "val" / "dummy_test_img_02.jpg"), dummy_img2)
        with open(dataset_root / "labels" / "val" / "dummy_test_img_02.txt", "w") as f:
            f.write(f"0 {55/320:.4f} {55/320:.4f} {50/320:.4f} {50/320:.4f}\n")

    print(f"Ensure your dataset path in '{dummy_yaml_path}' is correct or replace with your actual dataset YAML.")
    print("Using yolov8n.pt and yolov8n-cls.pt as dummy/test models.")

    try:
        validator = TwoStageValidator(
            dataset_yaml_path=str(dummy_yaml_path),
            stage1_model_path="yolov8n.pt", 
            stage1_model_arch="yolo",
            stage2_cls_model_path="yolov8n-cls.pt",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            imgsz=320,
            conf_thres_stage1=0.01, 
            iou_thres_matching=0.4
        )
        metrics = validator.validate()

        if metrics:
            print("\nReturned Metrics Summary:")
            for k, v_dict in metrics.items():
                if isinstance(v_dict, dict) and k not in ["s2_cls_stats_raw", "per_class_ap50", "per_class_ap75", "s2_accuracy_per_class"]:
                     print(f"  {k}: {v_dict}") # Should not happen with current structure
                elif isinstance(v_dict, dict): # For per_class dicts
                    print(f"  {k}:")
                    for k_sub, v_sub in v_dict.items():
                         print(f"    {k_sub}: {v_sub:.4f}" if isinstance(v_sub, float) else f"    {k_sub}: {v_sub}")
                else: # For simple float metrics
                    print(f"  {k}: {v_dict:.4f}" if isinstance(v_dict, float) else f"  {k}: {v_dict}")
    
    except FileNotFoundError as e:
        print(f"Setup Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during validation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass
