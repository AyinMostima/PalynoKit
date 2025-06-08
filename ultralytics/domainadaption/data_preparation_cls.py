# F:\Homework\MXERPRROJECT\FLOWER\ultralytics\domainadaption\data_preparation_cls.py
import logging
from pathlib import Path
import shutil
import random
from PIL import Image
import torch
from ultralytics import YOLO  # YOLO class is used for loading classification models too

# from torch.utils.data import Dataset # AddaDomainDataset will be imported or assumed available
# from torchvision import transforms as T # Used by AddaDomainDataset

# Assuming utils.py and AddaDomainDataset (from data_preparation.py) are accessible
# If running this directly, you might need to adjust imports or ensure domainadaption is in PYTHONPATH
try:
    from .utils import ensure_dir_exists, copy_files_from_list, get_image_files
    from .data_preparation import AddaDomainDataset  # Reusing this
except ImportError:
    # Fallback for direct execution if modules are not found in package structure
    # This assumes utils.py and data_preparation.py are in the same directory
    # For package use, the above try block should work.
    if __package__ is None or __package__ == '':
        import sys

        sys.path.append(str(Path(__file__).parent.parent))  # Go up to 'ultralytics' if needed
        from domainadaption.utils import ensure_dir_exists, copy_files_from_list, get_image_files
        from domainadaption.data_preparation import AddaDomainDataset
    else:
        raise

logger = logging.getLogger(__name__)


def get_class_subfolders(train_path: Path) -> list[Path]:
    """Lists all direct subdirectories (assumed to be class folders) in train_path."""
    if not train_path.is_dir():
        logger.error(f"Train path for classes not found: {train_path}")
        return []
    return [d for d in train_path.iterdir() if d.is_dir()]


def pseudo_label_target_images_cls(
        target_unclassified_img_dir: Path,
        num_target_images_to_sample: int,
        pretrained_cls_model_path: str,
        device: str = 'cpu'
) -> list[tuple[Path, str, float]]:
    """
    Performs pseudo-labeling for a sample of target domain classification images.

    Args:
        target_unclassified_img_dir: Path to the directory containing target domain images (e.g., test/).
        num_target_images_to_sample: Number of target images to randomly sample and pseudo-label.
        pretrained_cls_model_path: Path to the pretrained Ultralytics classification model (.pt).
        device: Device to run the model on ('cpu', 'cuda').

    Returns:
        A list of tuples: (image_path, predicted_class_name, confidence_score)
    """
    logger.info(
        f"Starting pseudo-labeling for {num_target_images_to_sample} target classification images from: {target_unclassified_img_dir}")
    logger.info(f"Using pretrained classification model: {pretrained_cls_model_path}")

    all_target_images = get_image_files(target_unclassified_img_dir)
    if not all_target_images:
        logger.warning(f"No images found in target directory: {target_unclassified_img_dir}")
        return []

    if len(all_target_images) < num_target_images_to_sample:
        logger.warning(f"Requested to sample {num_target_images_to_sample} images, "
                       f"but only {len(all_target_images)} available. Using all available.")
        images_to_label = all_target_images
    else:
        images_to_label = random.sample(all_target_images, num_target_images_to_sample)

    logger.info(f"Selected {len(images_to_label)} target images for pseudo-labeling.")

    try:
        model = YOLO(pretrained_cls_model_path)  # YOLO class handles classification models
        model.to(device)
        logger.info(f"Successfully loaded classification model: {pretrained_cls_model_path}")
    except Exception as e:
        logger.error(f"Failed to load classification model {pretrained_cls_model_path}: {e}", exc_info=True)
        return []

    pseudo_labeled_results = []
    processed_count = 0
    for img_path in images_to_label:
        try:
            results = model(str(img_path), verbose=False)  # Predict

            if results and results[0].probs is not None:
                top1_idx = results[0].probs.top1
                top1_conf = results[0].probs.top1conf.item()  # Get confidence of the top class
                predicted_class_name = results[0].names[top1_idx]

                pseudo_labeled_results.append((img_path, predicted_class_name, top1_conf))
                # logger.debug(f"Image: {img_path.name}, Predicted: {predicted_class_name}, Conf: {top1_conf:.4f}")
            else:
                logger.warning(f"Could not get probabilities for image: {img_path.name}")

            processed_count += 1
            if processed_count % 20 == 0 or processed_count == len(images_to_label):
                logger.info(f"Pseudo-labeled {processed_count}/{len(images_to_label)} target images.")

        except Exception as e:
            logger.error(f"Error during pseudo-labeling for image {img_path.name}: {e}", exc_info=True)

    logger.info(f"Pseudo-labeling finished. Successfully processed {len(pseudo_labeled_results)} images.")
    return pseudo_labeled_results


def create_adda_cls_paired_dataset(
        source_train_base_path: Path,  # Path to source domain's 'train' folder (e.g., .../test2/train/)
        pseudo_labeled_target_images: list[tuple[Path, str, float]],  # Output from pseudo_label_target_images_cls
        adda_cls_dataset_output_path: Path,  # Base path to create ADDA dataset (e.g., .../ADDAdatasets_cls)
        min_pseudo_label_conf: float = 0.0  # Minimum confidence for a pseudo-label to be used
):
    """
    Creates the ADDA classification dataset by pairing pseudo-labeled target images
    with source images of the same (pseudo-predicted) class.

    Args:
        source_train_base_path: Path to the source domain's 'train' directory, which contains class subfolders.
        pseudo_labeled_target_images: A list of (image_path, predicted_class_name, confidence) tuples.
        adda_cls_dataset_output_path: The base directory where 'images/source' and 'images/target' will be created.
        min_pseudo_label_conf: Threshold for using a pseudo-labeled target image.
    Returns:
        Tuple of (Path to ADDA source images, Path to ADDA target images)
    """
    logger.info("Creating paired ADDA classification dataset...")
    adda_source_img_output_dir = adda_cls_dataset_output_path / 'images' / 'source'
    adda_target_img_output_dir = adda_cls_dataset_output_path / 'images' / 'target'

    ensure_dir_exists(adda_source_img_output_dir)
    ensure_dir_exists(adda_target_img_output_dir)

    copied_source_count = 0
    copied_target_count = 0
    skipped_low_conf_count = 0

    if not pseudo_labeled_target_images:
        logger.warning("No pseudo-labeled target images provided. ADDA dataset will be empty.")
        return adda_source_img_output_dir, adda_target_img_output_dir

    for target_img_path, predicted_class_name, conf in pseudo_labeled_target_images:
        if conf < min_pseudo_label_conf:
            skipped_low_conf_count += 1
            # logger.debug(f"Skipping target image {target_img_path.name} due to low confidence ({conf:.2f} < {min_pseudo_label_conf}).")
            continue

        # 1. Copy target image
        try:
            target_dest_path = adda_target_img_output_dir / target_img_path.name
            if not target_dest_path.exists():  # Avoid re-copying if names clash, though unlikely with random sampling
                shutil.copy(target_img_path, target_dest_path)
                copied_target_count += 1
            else:
                logger.warning(f"Target image {target_img_path.name} already exists in destination. Skipping copy.")

        except Exception as e:
            logger.error(f"Error copying target image {target_img_path} to {adda_target_img_output_dir}: {e}")
            continue  # Skip this pair if target can't be copied

        # 2. Find and copy a corresponding source image
        source_class_folder = source_train_base_path / predicted_class_name
        if not source_class_folder.is_dir():
            logger.warning(f"Source class folder '{predicted_class_name}' not found at '{source_class_folder}'. "
                           f"Cannot find matching source image for {target_img_path.name}.")
            continue

        available_source_images_for_class = get_image_files(source_class_folder)
        if not available_source_images_for_class:
            logger.warning(f"No source images found in class folder '{source_class_folder}' "
                           f"for predicted class '{predicted_class_name}'.")
            continue

        # Randomly select one source image from that class
        selected_source_img_path = random.choice(available_source_images_for_class)
        try:
            # To avoid name clashes if multiple target images map to the same source class,
            # and we pick the same random source image by chance (unlikely for many images),
            # or if different source classes have images with same names (bad practice).
            # A simple way is to use target image's stem for the source copy name to ensure pairing.
            source_dest_filename = f"source_for_{target_img_path.stem}{selected_source_img_path.suffix}"
            source_dest_path = adda_source_img_output_dir / source_dest_filename

            shutil.copy(selected_source_img_path, source_dest_path)
            copied_source_count += 1
        except Exception as e:
            logger.error(f"Error copying source image {selected_source_img_path} to {adda_source_img_output_dir}: {e}")
            # If source copy fails, we might have an orphaned target image in the ADDA set.
            # Depending on strictness, one might remove the copied target image here.
            # For now, we log and continue.

    if skipped_low_conf_count > 0:
        logger.info(f"Skipped {skipped_low_conf_count} target images due to low pseudo-label confidence.")
    logger.info(f"ADDA (paired) classification dataset created: "
                f"{copied_source_count} source images, {copied_target_count} target images.")
    logger.info(f"  Source images at: {adda_source_img_output_dir}")
    logger.info(f"  Target images at: {adda_target_img_output_dir}")

    return adda_source_img_output_dir, adda_target_img_output_dir


if __name__ == '__main__':
    # Ensure the main domainadaption logger is set up if running standalone for testing
    if __package__ is None or __package__ == '':
        # If run as a script, set up a basic logger for the current module.
        # For package usage, the main run script should configure logging.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("Testing data_preparation_cls.py...")

    # --- Test Configuration ---
    # User should create this structure and populate with a few images
    # I:\pollendataset\domainadaptionval\test2_example\
    # ├── train\
    # │   ├── airplane\
    # │   │   ├── plane1.jpg
    # │   │   └── plane2.png
    # │   └── car\
    # │       ├── car1.jpg
    # │       └── car2.png
    # └── test\  (target, unclassified)
    #     ├── target_img1.jpg
    #     ├── target_img2.png
    #     └── target_img3.bmp

    TEST_CLS_BASE_DATA_PATH = Path(r'I:\pollendataset\domainadaptionval\test2_example_dp_cls')
    TEST_CLS_BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Create dummy structure and files if they don't exist for the test
    (TEST_CLS_BASE_DATA_PATH / "train" / "airplane").mkdir(parents=True, exist_ok=True)
    (TEST_CLS_BASE_DATA_PATH / "train" / "car").mkdir(parents=True, exist_ok=True)
    (TEST_CLS_BASE_DATA_PATH / "test").mkdir(parents=True, exist_ok=True)

    try:
        Image.new('RGB', (60, 30), color='lightblue').save(
            TEST_CLS_BASE_DATA_PATH / "train" / "airplane" / "plane1.jpg")
        Image.new('RGB', (60, 30), color='skyblue').save(TEST_CLS_BASE_DATA_PATH / "train" / "airplane" / "plane2.png")
        Image.new('RGB', (60, 30), color='lightcoral').save(TEST_CLS_BASE_DATA_PATH / "train" / "car" / "car1.jpg")
        Image.new('RGB', (60, 30), color='salmon').save(TEST_CLS_BASE_DATA_PATH / "train" / "car" / "car2.png")
        Image.new('RGB', (60, 30), color='grey').save(TEST_CLS_BASE_DATA_PATH / "test" / "target_img1.jpg")
        Image.new('RGB', (60, 30), color='darkgrey').save(TEST_CLS_BASE_DATA_PATH / "test" / "target_img2.png")
        logger.info(f"Created dummy files in {TEST_CLS_BASE_DATA_PATH}")
    except Exception as e:
        logger.warning(f"Could not create all dummy files for testing: {e}")

    # Path to a pretrained Ultralytics classification model (e.g., yolov8n-cls.pt)
    # Download if you don't have one:
    # import ultralytics; ultralytics.checks() # often downloads default models
    PRETRAINED_CLS_MODEL = "yolov8n-cls.pt"  # Replace if needed

    ADDA_CLS_OUTPUT_PATH = TEST_CLS_BASE_DATA_PATH.parent / "ADDA_CLS_Output_Test"
    ADDA_CLS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    NUM_TARGET_TO_SAMPLE_FOR_PSEUDO = 2  # Number of target images to pseudo-label

    # 1. Test pseudo-labeling
    logger.info("\n--- Testing Pseudo-Labeling for Classification ---")
    if Path(PRETRAINED_CLS_MODEL).exists() and (TEST_CLS_BASE_DATA_PATH / "test").exists():
        pseudo_labeled_targets = pseudo_label_target_images_cls(
            target_unclassified_img_dir=TEST_CLS_BASE_DATA_PATH / "test",
            num_target_images_to_sample=NUM_TARGET_TO_SAMPLE_FOR_PSEUDO,
            pretrained_cls_model_path=PRETRAINED_CLS_MODEL,
            device='cpu'
        )
        if pseudo_labeled_targets:
            logger.info(f"Pseudo-labeled target results: {pseudo_labeled_targets}")

            # 2. Test ADDA (paired) dataset creation
            logger.info("\n--- Testing ADDA Paired Classification Dataset Creation ---")
            if (TEST_CLS_BASE_DATA_PATH / "train").exists():
                adda_source_dir, adda_target_dir = create_adda_cls_paired_dataset(
                    source_train_base_path=TEST_CLS_BASE_DATA_PATH / "train",
                    pseudo_labeled_target_images=pseudo_labeled_targets,
                    adda_cls_dataset_output_path=ADDA_CLS_OUTPUT_PATH,
                    min_pseudo_label_conf=0.1  # Example confidence threshold
                )
                logger.info(f"ADDA source images for classification created at: {adda_source_dir}")
                logger.info(f"ADDA target images for classification created at: {adda_target_dir}")

                # 3. Test reusable AddaDomainDataset with the new cls data
                if adda_source_dir.exists() and adda_target_dir.exists() and \
                        len(list(adda_source_dir.glob("*"))) > 0:
                    logger.info("\n--- Testing AddaDomainDataset with Classification ADDA Data ---")
                    source_cls_adda_dataset = AddaDomainDataset(
                        img_dir=adda_source_dir,
                        img_size=224,  # Typical for classification
                        domain_label=1.0  # SOURCE_DOMAIN_LABEL
                    )
                    if len(source_cls_adda_dataset) > 0:
                        img, label = source_cls_adda_dataset[0]
                        logger.info(f"Sample from source ADDA CLS dataset - Img shape: {img.shape}, Label: {label}")
                    else:
                        logger.warning("Source ADDA CLS dataset is empty.")
                else:
                    logger.warning("Could not test AddaDomainDataset due to empty ADDA cls data folders.")
            else:
                logger.warning(
                    f"Source train path {TEST_CLS_BASE_DATA_PATH / 'train'} not found. Skipping ADDA paired dataset creation.")
        else:
            logger.warning("Pseudo-labeling did not return any results. Skipping paired dataset creation.")
    else:
        logger.warning(f"Pretrained classification model '{PRETRAINED_CLS_MODEL}' not found or "
                       f"target test directory '{TEST_CLS_BASE_DATA_PATH / 'test'}' not found. Skipping tests.")

    logger.info(f"\nTest run of data_preparation_cls.py finished. Check outputs in {ADDA_CLS_OUTPUT_PATH}")
    # For cleanup: shutil.rmtree(TEST_CLS_BASE_DATA_PATH)
    # For cleanup: shutil.rmtree(ADDA_CLS_OUTPUT_PATH)
