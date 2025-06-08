import logging
from pathlib import Path
import torch

# --- Import from your ADDA package ---
from ultralytics.domainadaption.utils import setup_logging, get_device
from ultralytics.domainadaption.data_preparation_cls import (
    pseudo_label_target_images_cls,
    create_adda_cls_paired_dataset
)
# Reusing AddaDomainDataset from the original data_preparation.py
from ultralytics.domainadaption.data_preparation import AddaDomainDataset

# Model setup, discriminator, and trainer would be similar but geared for CLS models
# For instance, create_adda_feature_extractors would load YOLO('your_cls_model.pt')
# The adapt_and_hook_layer_indices would refer to layers in a CLS model.
from ultralytics.domainadaption.model_setup import (
    create_adda_feature_extractors, # Needs to correctly load CLS models
    determine_feature_dim_for_discriminator,
    save_adapted_ultralytics_model, # Needs to correctly save CLS models
    FEATURES_STORAGE
)
from ultralytics.domainadaption.discriminator_architectures import get_discriminator
from ultralytics.domainadaption.adda_trainer import train_adda # Trainer logic is general

logger = logging.getLogger(__name__)

def run_complete_adda_cls_process(
    # --- Path Configurations ---
    base_dataset_path_cls: Path,  # Path to CLS dataset (e.g., test2/ with train/classA, test/imgB)
    experiment_base_path: Path,
    pretrained_cls_model_pt_path: str, # Path to the .pt classification model

    # --- Data Preparation Parameters ---
    num_target_images_for_pseudo: int = 100,
    min_pseudo_label_confidence: float = 0.5,

    # --- Model Adaptation Parameters ---
    adapt_and_hook_layer_indices_cls: list[int]=[0,2,4], # Indices for CLS model

    # --- ADDA Training Parameters ---
    img_size_adda_cls: int = 224, # Common for classification
    batch_size_adda_cls: int = 16,
    learning_rate_f_t_cls: float = 1e-5,
    learning_rate_d_cls: float = 1e-4,
    num_epochs_adda_cls: int = 20,

    device_str: str = "auto",
    run_pseudo_labeling_cls: bool = True
):
    experiment_name = f"ADDA_CLS_run_{Path(pretrained_cls_model_pt_path).stem}_{'_'.join(map(str, adapt_and_hook_layer_indices_cls))}"
    current_experiment_path = experiment_base_path / experiment_name
    current_experiment_path.mkdir(parents=True, exist_ok=True)

    log_file = current_experiment_path / "adda_cls_process_log.txt"
    setup_logging(log_file_path=log_file)
    device = get_device(device_str)

    logger.info(f"Starting ADDA CLASSIFICATION process: {experiment_name}")
    logger.info(f"Output path: {current_experiment_path}")
    logger.info(f"Pretrained source CLS model: {pretrained_cls_model_pt_path}")
    logger.info(f"Layers to adapt/hook in CLS model: {adapt_and_hook_layer_indices_cls}")

    # --- 1. Data Preparation for Classification ADDA ---
    source_train_path = base_dataset_path_cls / "train"
    target_test_images_path = base_dataset_path_cls / "test"

    pseudo_labeled_target_list = []
    if run_pseudo_labeling_cls:
        logger.info("\n--- Step 1.a: Pseudo-Labeling Target Classification Images ---")
        if not target_test_images_path.exists():
            logger.error(f"Target test images directory not found: {target_test_images_path}")
            return
        pseudo_labeled_target_list = pseudo_label_target_images_cls(
            target_unclassified_img_dir=target_test_images_path,
            num_target_images_to_sample=num_target_images_for_pseudo,
            pretrained_cls_model_path=pretrained_cls_model_pt_path,
            device=device.type
        )
        if not pseudo_labeled_target_list:
            logger.error("Pseudo-labeling yielded no results. Cannot proceed with paired dataset creation.")
            return
    else:
        logger.info("Skipping pseudo-labeling for target classification images.")
        # If not running pseudo-labeling, this example assumes you have another way
        # to prepare the target images for the ADDA classification dataset.
        # For the paired approach, pseudo_labeled_target_list is essential.
        # For a simpler ADDA (random source, random target), you'd modify create_adda_cls_paired_dataset
        # or use a different function. Sticking to the "paired" request for now.
        logger.warning("Pseudo-labeling skipped. The current 'create_adda_cls_paired_dataset' relies on pseudo-labels.")
        logger.warning("If you intend to use target images without pseudo-labels for pairing, adjust logic or provide them differently.")
        # For this example to proceed, we need pseudo_labeled_target_list. If not running, this will fail later.
        # A robust script might handle this by trying to load pre-existing pseudo_labels or erroring out.
        if not pseudo_labeled_target_list: # if still empty
             logger.error("Pseudo-labeling was skipped and no pseudo_labeled_target_list provided. Aborting.")
             return


    logger.info("\n--- Step 1.b: Creating Paired ADDA Classification Dataset ---")
    adda_cls_dataset_output_base = current_experiment_path / 'ADDA_paired_cls_dataset'
    if not source_train_path.exists():
        logger.error(f"Source train directory for classes not found: {source_train_path}")
        return

    # This function creates 'images/source' and 'images/target' under adda_cls_dataset_output_base
    adda_source_img_dir, adda_target_img_dir = create_adda_cls_paired_dataset(
        source_train_base_path=source_train_path,
        pseudo_labeled_target_images=pseudo_labeled_target_list,
        adda_cls_dataset_output_path=adda_cls_dataset_output_base,
        min_pseudo_label_conf=min_pseudo_label_confidence
    )

    if not list(adda_source_img_dir.glob("*")) or not list(adda_target_img_dir.glob("*")):
        logger.error(f"ADDA source or target image directory is empty after creation. Check logs. Path S: {adda_source_img_dir}, Path T: {adda_target_img_dir}")
        return

    # --- 2. Model Setup (F_s_cls, F_t_cls) ---
    # create_adda_feature_extractors needs to be able to load classification models correctly.
    # The existing one tries YOLO then RTDETR. YOLO(model.pt) should load cls models.
    logger.info("\n--- Step 2: Setting up F_s_cls, F_t_cls Feature Extractors ---")
    try:
        F_s_cls, F_t_cls, adaptable_params_F_t_cls, original_cls_model_wrapper = create_adda_feature_extractors(
            pretrained_model_path=pretrained_cls_model_pt_path,
            adapt_layer_indices=adapt_and_hook_layer_indices_cls,
            device_str=device.type
        )
    except Exception as e:
        logger.error(f"Error creating CLS feature extractors: {e}", exc_info=True)
        return

    # --- 3. Determine Discriminator Input Dimension ---
    logger.info("\n--- Step 3: Determining Discriminator Input Dimension for CLS features ---")
    try:
        discriminator_input_dim_cls = determine_feature_dim_for_discriminator(
            model_for_dim_check=F_s_cls,
            hook_target_layer_indices=adapt_and_hook_layer_indices_cls,
            img_size=img_size_adda_cls, # Use CLS ADDA image size
            device_str=device.type
        )
    except Exception as e:
        logger.error(f"Error determining CLS discriminator input dimension: {e}", exc_info=True)
        return

    # --- 4. Create Discriminator ---
    logger.info("\n--- Step 4: Creating Domain Discriminator for CLS features ---")
    try:
        D_cls = get_discriminator(
            name="default",
            input_dim=discriminator_input_dim_cls,
            hidden_dim=512 # Example for CLS
        ).to(device)
    except Exception as e:
        logger.error(f"Error creating CLS discriminator: {e}", exc_info=True)
        return

    # --- 5. Run ADDA Training for Classification Models ---
    logger.info("\n--- Step 5: Starting ADDA Training Loop for CLS Models ---")
    try:
        F_t_cls_adapted = train_adda(
            F_s=F_s_cls,
            F_t=F_t_cls,
            D=D_cls,
            adaptable_params_F_t=adaptable_params_F_t_cls,
            hook_layer_indices_for_adda=adapt_and_hook_layer_indices_cls,
            source_adda_img_dir=adda_source_img_dir, # From create_adda_cls_paired_dataset
            target_adda_img_dir=adda_target_img_dir, # From create_adda_cls_paired_dataset
            experiment_path=current_experiment_path,
            img_size=img_size_adda_cls,
            batch_size=batch_size_adda_cls,
            learning_rate_t=learning_rate_f_t_cls,
            learning_rate_d=learning_rate_d_cls,
            num_epochs=num_epochs_adda_cls,
            device_str=device.type,
            save_intermediate_models=True
        )
    except Exception as e:
        logger.error(f"Critical error during CLS ADDA training: {e}", exc_info=True)
        return

    # --- 6. Save Final Adapted Classification Model ---
    logger.info("\n--- Step 6: Saving Final Adapted Ultralytics CLS Model ---")
    final_adapted_cls_model_path = current_experiment_path / f"{Path(pretrained_cls_model_pt_path).stem}_ADDA_CLS_adapted.pt"
    try:
        save_adapted_ultralytics_model(
            original_model_wrapper=original_cls_model_wrapper,
            adapted_F_t_state_dict=F_t_cls_adapted.state_dict(),
            adapt_layer_indices=adapt_and_hook_layer_indices_cls,
            output_pt_path=final_adapted_cls_model_path,
            device_str=device.type
        )
        logger.info(f"Successfully saved final ADDA-adapted CLS model to: {final_adapted_cls_model_path}")
    except Exception as e:
        logger.error(f"Error saving final adapted CLS model: {e}", exc_info=True)
        fallback_path = current_experiment_path / f"{Path(pretrained_cls_model_pt_path).stem}_ADDA_CLS_adapted_Ft_statedict_only.pt"
        torch.save(F_t_cls_adapted.state_dict(), fallback_path)
        logger.info(f"Saved F_t_cls_adapted state_dict directly to: {fallback_path}")

    logger.info(f"\nADDA CLASSIFICATION process finished: {experiment_name}")

