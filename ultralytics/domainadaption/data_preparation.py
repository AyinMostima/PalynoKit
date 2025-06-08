import logging
from pathlib import Path
import shutil
import random
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO, RTDETR
from torch.utils.data import Dataset
from torchvision import transforms as T

from .utils import ensure_dir_exists, copy_files_from_list, get_image_files, SOURCE_DOMAIN_LABEL, TARGET_DOMAIN_LABEL

logger = logging.getLogger(__name__)


def _load_model_for_pseudo_labeling(pretrained_model_path: str, device: str, model_choice: str):
    """
    Loads the pretrained model based on the specified model_choice ('yolo' or 'rtdetr').
    Returns the loaded model instance ready for prediction, or None if loading fails.
    """
    model_instance = None
    choice = model_choice.lower()
    logger.info(f"Attempting to load pseudo-labeling model: {pretrained_model_path} as {choice.upper()}")

    if choice == 'yolo':
        try:
            model_instance = YOLO(pretrained_model_path)
            logger.info(f"Successfully loaded model as YOLO: {pretrained_model_path}")
        except Exception as e_yolo:
            logger.error(f"Failed to load model {pretrained_model_path} as YOLO: {e_yolo}.")
            return None
    elif choice == 'rtdetr':
        try:
            model_instance = RTDETR(pretrained_model_path)
            logger.info(f"Successfully loaded model as RTDETR: {pretrained_model_path}")
        except Exception as e_rtdetr:
            logger.error(f"Failed to load model {pretrained_model_path} as RTDETR: {e_rtdetr}.")
            return None
    else:
        logger.error(f"Invalid model_choice: '{model_choice}'. Must be 'yolo' or 'rtdetr'.")
        return None

    if model_instance:
        model_instance.to(device)
    return model_instance


def pseudo_label_target_domain(
        target_image_dir: Path,
        output_pseudo_label_dir: Path,
        pretrained_model_path: str,
        model_load_choice: str,  # **新的参数，用于选择加载 YOLO 还是 RTDETR**
        img_size: int = 640,
        confidence_threshold: float = 0.25,
        device: str = 'cpu'
):
    """
    Generates pseudo-labels for images in the target domain using a pretrained model.
    Saves labels in YOLO format to output_pseudo_label_dir.
    """
    ensure_dir_exists(output_pseudo_label_dir)
    logger.info(f"Starting pseudo-labeling for target images in: {target_image_dir}")
    logger.info(f"Using pretrained model: {pretrained_model_path} (attempting to load as **{model_load_choice.upper()}**)")
    logger.info(f"Outputting pseudo-labels to: {output_pseudo_label_dir}")

    # **使用 model_load_choice 参数调用 _load_model_for_pseudo_labeling**
    model = _load_model_for_pseudo_labeling(pretrained_model_path, device, model_load_choice)

    if model is None:
        logger.error(
            f"Failed to load pretrained model for pseudo-labeling: {pretrained_model_path} with choice {model_load_choice}. Aborting pseudo-labeling.")
        return

    target_image_files = get_image_files(target_image_dir)
    if not target_image_files:
        logger.warning(f"No images found in target directory: {target_image_dir}")
        return

    processed_count = 0
    for img_path in target_image_files:
        try:
            results = model.predict(source=str(img_path), imgsz=img_size, conf=confidence_threshold, device=device,
                                    verbose=False)

            label_file_path = output_pseudo_label_dir / (img_path.stem + ".txt")
            with open(label_file_path, 'w') as f:
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes_xywhn = results[0].boxes.xywhn.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    for cls_id, box_coords in zip(classes, boxes_xywhn):
                        x_center, y_center, width, height = box_coords
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            processed_count += 1
            if processed_count % 50 == 0 or processed_count == len(target_image_files):
                logger.info(f"Processed {processed_count}/{len(target_image_files)} images for pseudo-labeling.")
        except Exception as e:
            logger.error(f"Error processing image {img_path} for pseudo-labeling: {e}", exc_info=True)

    logger.info(f"Pseudo-labeling complete. Processed {processed_count} images.")
    logger.info(f"Pseudo-labels saved in: {output_pseudo_label_dir}")


def create_adda_classification_dataset_folders(
        base_dataset_path: Path,
        adda_dataset_base_path: Path,
        num_images_per_domain: int = 500
):
    """
    Creates the directory structure for the ADDA binary classification task.
    Copies images from source (train split) and target (val split) domains
    from the base_dataset_path to the adda_dataset_base_path.
    """
    logger.info(f"Creating ADDA classification dataset folders using base: {base_dataset_path}")
    logger.info(f"Outputting ADDA classification images to: {adda_dataset_base_path / 'images'}")

    adda_source_img_output_path = adda_dataset_base_path / 'images' / 'source'
    adda_target_img_output_path = adda_dataset_base_path / 'images' / 'target'

    ensure_dir_exists(adda_source_img_output_path)
    ensure_dir_exists(adda_target_img_output_path)

    source_train_img_dir = base_dataset_path / 'images' / 'train'
    source_train_label_dir = base_dataset_path / 'labels' / 'train'
    target_val_img_dir = base_dataset_path / 'images' / 'val'

    if not source_train_img_dir.is_dir():
        logger.error(f"Source domain training image directory not found: {source_train_img_dir}")
        raise FileNotFoundError(f"Source domain training image directory not found: {source_train_img_dir}")
    if not source_train_label_dir.is_dir():
        logger.error(f"Source domain training label directory not found: {source_train_label_dir}")
        raise FileNotFoundError(f"Source domain training label directory not found: {source_train_label_dir}")
    if not target_val_img_dir.is_dir():
        logger.error(f"Target domain validation image directory not found: {target_val_img_dir}")
        raise FileNotFoundError(f"Target domain validation image directory not found: {target_val_img_dir}")

    source_img_files = get_image_files(source_train_img_dir)
    target_img_files = get_image_files(target_val_img_dir)

    num_source_to_copy = min(num_images_per_domain, len(source_img_files))
    num_target_to_copy = min(num_images_per_domain, len(target_img_files))

    if len(source_img_files) < num_images_per_domain:
        logger.warning(
            f"Requested {num_images_per_domain} source images, but only {len(source_img_files)} available. Using all {len(source_img_files)}.")
    if len(target_img_files) < num_images_per_domain:
        logger.warning(
            f"Requested {num_images_per_domain} target images, but only {len(target_img_files)} available. Using all {len(target_img_files)}.")

    if not source_img_files: logger.warning(f"No source images found to copy from {source_train_img_dir}.")
    if not target_img_files: logger.warning(f"No target images found to copy from {target_val_img_dir}.")

    logger.info(
        f"Selecting {num_source_to_copy} source images (from train split) and {num_target_to_copy} target images (from val split) for ADDA classification dataset.")

    copy_files_from_list(source_img_files, adda_source_img_output_path, num_source_to_copy)
    copy_files_from_list(target_img_files, adda_target_img_output_path, num_target_to_copy)

    logger.info(f"ADDA classification image dataset structure created at: {adda_dataset_base_path / 'images'}")
    return adda_source_img_output_path, adda_target_img_output_path


class AddaDomainDataset(Dataset):
    def __init__(self, img_dir: Path, img_size: int, domain_label: float, transform=None):
        self.img_dir = Path(img_dir)
        self.img_files = get_image_files(self.img_dir)
        self.img_size = img_size
        self.domain_label = torch.tensor([domain_label], dtype=torch.float32)

        if transform is None:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        if not self.img_files:
            logger.warning(f"No images found in {self.img_dir} for AddaDomainDataset (label: {domain_label}).")
        else:
            logger.info(
                f"Created AddaDomainDataset for {self.img_dir} with {len(self.img_files)} images. Domain label: {domain_label}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
        except Exception as e:
            logger.error(f"Error loading or transforming image {img_path}: {e}. Returning zero tensor.")
            img_tensor = torch.zeros((3, self.img_size, self.img_size))

        return img_tensor, self.domain_label


if __name__ == '__main__':
    from .utils import setup_logging

    setup_logging(level=logging.DEBUG)

    TEST_PROJECT_ROOT = Path(r'F:\Homework\MXERPRROJECT\FLOWER\ultralytics\domainadaptiontest\DP_TEST_Unified')
    TEST_PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    TEST_BASE_DATA_PATH = TEST_PROJECT_ROOT / 'unified_dataset'
    (TEST_BASE_DATA_PATH / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (TEST_BASE_DATA_PATH / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (TEST_BASE_DATA_PATH / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (TEST_BASE_DATA_PATH / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    try:
        Image.new('RGB', (100, 100), color='red').save(TEST_BASE_DATA_PATH / 'images' / 'train' / 's_img1.jpg')
        with open(TEST_BASE_DATA_PATH / 'labels' / 'train' / 's_img1.txt', 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        Image.new('RGB', (100, 100), color='blue').save(TEST_BASE_DATA_PATH / 'images' / 'val' / 't_img1.jpg')
        logger.info(f"Created dummy data structure under {TEST_BASE_DATA_PATH}")
    except Exception as e:
        logger.error(f"Could not create dummy files for testing: {e}")

    TARGET_VAL_IMAGE_DIR_FOR_PSEUDO = TEST_BASE_DATA_PATH / 'images' / 'val'
    PSEUDO_LABEL_OUTPUT_DIR = TEST_PROJECT_ROOT / 'pseudo_labels_output_unified'

    YOLO_MODEL_PATH = "yolov8n.pt"
    # **假设 yolov8n.pt 存在且为 YOLO 模型**
    # **如果测试 RTDETR，确保 RTDETR_MODEL_PATH 指向一个有效的 RT-DETR .pt 文件**
    # RTDETR_MODEL_PATH = "rtdetr-l.pt" # 示例

    if Path(YOLO_MODEL_PATH).exists() and len(list(TARGET_VAL_IMAGE_DIR_FOR_PSEUDO.glob('*'))) > 0:
        logger.info(f"\n--- Testing Pseudo-Labeling with YOLO model (Unified Structure): {YOLO_MODEL_PATH} ---")
        pseudo_label_target_domain(
            target_image_dir=TARGET_VAL_IMAGE_DIR_FOR_PSEUDO,
            output_pseudo_label_dir=PSEUDO_LABEL_OUTPUT_DIR / "yolo_pseudo",
            pretrained_model_path=YOLO_MODEL_PATH,
            model_load_choice="yolo",  # **指定加载类型为 'yolo'**
            img_size=320,
            confidence_threshold=0.25,
            device='cpu'
        )
    else:
        logger.warning(f"Skipping YOLO pseudo-labeling test. Model {YOLO_MODEL_PATH} or target images in {TARGET_VAL_IMAGE_DIR_FOR_PSEUDO} missing.")

    # **可以为 RTDETR 添加类似的测试 (如果需要)**
    # if Path(RTDETR_MODEL_PATH).exists() and len(list(TARGET_VAL_IMAGE_DIR_FOR_PSEUDO.glob('*'))) > 0:
    #     logger.info(f"\n--- Testing Pseudo-Labeling with RTDETR model (Unified Structure): {RTDETR_MODEL_PATH} ---")
    #     pseudo_label_target_domain(
    #         target_image_dir=TARGET_VAL_IMAGE_DIR_FOR_PSEUDO,
    #         output_pseudo_label_dir=PSEUDO_LABEL_OUTPUT_DIR / "rtdetr_pseudo",
    #         pretrained_model_path=RTDETR_MODEL_PATH,
    #         model_load_choice="rtdetr",  # **指定加载类型为 'rtdetr'**
    #         img_size=320,
    #         confidence_threshold=0.25,
    #         device='cpu'
    #     )
    # else:
    #     logger.warning(f"Skipping RTDETR pseudo-labeling test. Model or target images missing.")


    ADDA_CLF_OUTPUT_BASE_PATH = TEST_PROJECT_ROOT / 'ADDA_classification_dataset_unified'
    if len(list((TEST_BASE_DATA_PATH / 'images' / 'train').glob('*'))) > 0 and \
            len(list((TEST_BASE_DATA_PATH / 'images' / 'val').glob('*'))) > 0:
        logger.info("\n--- Testing ADDA Classification Dataset Folder Creation (Unified Structure) ---")
        created_adda_source_path, created_adda_target_path = create_adda_classification_dataset_folders(
            base_dataset_path=TEST_BASE_DATA_PATH,
            adda_dataset_base_path=ADDA_CLF_OUTPUT_BASE_PATH,
            num_images_per_domain=1
        )
        if created_adda_source_path.exists() and len(list(created_adda_source_path.glob("*"))) > 0:
            logger.info("\n--- Testing AddaDomainDataset (Unified Structure) ---")
            source_adda_ds = AddaDomainDataset(
                img_dir=created_adda_source_path,
                img_size=320,
                domain_label=SOURCE_DOMAIN_LABEL
            )
            if len(source_adda_ds) > 0:
                img, label = source_adda_ds[0]
                logger.info(f"Source ADDA dataset sample - Img shape: {img.shape}, Label: {label}")
            else:
                logger.warning("Source ADDA dataset is empty after creation.")
        else:
            logger.warning(f"Path for source ADDA dataset images not found or empty: {created_adda_source_path}")
    else:
        logger.warning(
            "Skipping ADDA classification dataset creation test due to missing dummy source/target images in base path.")

    logger.info("data_preparation.py example usage with unified structure finished.")

