
import logging
from pathlib import Path
import torch # For torch.save if needed directly, and type hints
from PIL import Image # For creating dummy images in the example

# --- Import from your ADDA package ---
from .domainadaption.utils import setup_logging, get_device
from .domainadaption.data_preparation import pseudo_label_target_domain, create_adda_classification_dataset_folders
from .domainadaption.model_setup import (create_adda_feature_extractors, determine_feature_dim_for_discriminator,
                                        save_adapted_ultralytics_model, FEATURES_STORAGE)
from .domainadaption.discriminator_architectures import get_discriminator
from .domainadaption.adda_trainer import train_adda

logger = logging.getLogger(__name__)


def run_complete_adda_process(
    # --- Path Configurations ---
    base_dataset_path: Path,      # Unified path to dataset (images/train, images/val, labels/train, etc.)
    experiment_base_path: Path,   # Base path for all outputs of this ADDA run
    pretrained_model_pt_path: str,# Path to the .pt file of the model trained on source domain

    # --- Data Preparation Parameters ---
    num_images_for_adda_cls: int = 500,
    pseudo_label_conf_thresh: float = 0.25,

    # --- Model Adaptation Parameters ---
    adapt_and_hook_layer_indices: list[int] = [10, 12, 14],

    # --- ADDA Training Parameters ---
    img_size_adda: int = 640,
    batch_size_adda: int = 4,
    learning_rate_f_t: float = 1e-5,
    learning_rate_d: float = 1e-4,
    num_epochs_adda: int = 10,

    # --- System Configuration ---
    device_str: str = "auto",
    run_pseudo_labeling: bool = True
):
    """
    Main function to orchestrate the entire ADDA process using a unified dataset path.
    """
    # --- 0. Initial Setup ---
    experiment_name = f"ADDA_run_{Path(pretrained_model_pt_path).stem}_{Path(base_dataset_path).name}_{'_'.join(map(str, adapt_and_hook_layer_indices))}"
    current_experiment_path = experiment_base_path / experiment_name
    current_experiment_path.mkdir(parents=True, exist_ok=True)

    log_file = current_experiment_path / "adda_process_log.txt"
    setup_logging(log_file_path=log_file, level=logging.INFO) # Ensure INFO level
    device = get_device(device_str)

    logger.info(f"Starting ADDA process for experiment: {experiment_name}")
    logger.info(f"Unified dataset path: {base_dataset_path}")
    logger.info(f"Experiment output path: {current_experiment_path}")
    logger.info(f"Using device: {device}")
    logger.info(f"Pretrained source model: {pretrained_model_pt_path}")
    logger.info(f"Layers to adapt and hook for features: {adapt_and_hook_layer_indices}")

    # --- 1. Data Preparation ---
    # Define specific paths from base_dataset_path
    target_val_images_dir = base_dataset_path / "images" / "val"
    # target_val_labels_dir = base_dataset_path / "labels" / "val" # Optional, for reference

    # 1.a. Pseudo-label target domain
    target_pseudo_labels_dir = current_experiment_path / "target_pseudo_labels"
    if run_pseudo_labeling:
        logger.info("\n--- Step 1.a: Pseudo-Labeling Target Domain (from val split) ---")
        if not target_val_images_dir.is_dir() or not any(target_val_images_dir.iterdir()):
            logger.warning(f"Target validation image directory for pseudo-labeling is empty or does not exist: {target_val_images_dir}. Skipping pseudo-labeling.")
        else:
            pseudo_label_target_domain(
                target_image_dir=target_val_images_dir, # Use the specific target images directory
                output_pseudo_label_dir=target_pseudo_labels_dir,
                pretrained_model_path=pretrained_model_pt_path,
                img_size=img_size_adda,
                confidence_threshold=pseudo_label_conf_thresh,
                device=device.type
            )
    else:
        logger.info("Skipping pseudo-labeling step for target domain.")

    # 1.b. Create ADDA binary classification dataset (source vs target images)
    logger.info("\n--- Step 1.b: Creating ADDA Classification Dataset Folders ---")
    adda_clf_dataset_path = current_experiment_path / 'adda_classification_dataset'

    # create_adda_classification_dataset_folders now takes base_dataset_path
    # and internally derives source (train) and target (val) image paths.
    try:
        adda_source_img_dir_for_loader, adda_target_img_dir_for_loader = create_adda_classification_dataset_folders(
            base_dataset_path=base_dataset_path, # Pass the unified base path
            adda_dataset_base_path=adda_clf_dataset_path,
            num_images_per_domain=num_images_for_adda_cls
        )
    except FileNotFoundError as e:
        logger.error(f"Critical error during ADDA classification dataset creation: {e}. Aborting.")
        return
    except Exception as e:
        logger.error(f"Unexpected error during ADDA classification dataset creation: {e}. Aborting.", exc_info=True)
        return


    # --- 2. Model Setup ---
    logger.info("\n--- Step 2: Setting up F_s, F_t Feature Extractors ---")
    try:
        F_s, F_t, adaptable_params_F_t, original_model_wrapper = create_adda_feature_extractors(
            pretrained_model_path=pretrained_model_pt_path,
            adapt_layer_indices=adapt_and_hook_layer_indices,
            device_str=device.type
        )
    except Exception as e:
        logger.error(f"Critical error setting up feature extractors: {e}. Aborting.", exc_info=True)
        return

    # --- 3. Determine Discriminator Input Dimension ---
    logger.info("\n--- Step 3: Determining Discriminator Input Dimension ---")
    try:
        discriminator_input_dim = determine_feature_dim_for_discriminator(
            model_for_dim_check=F_s,
            hook_target_layer_indices=adapt_and_hook_layer_indices,
            img_size=img_size_adda,
            device_str=device.type
        )
    except Exception as e:
        logger.error(f"Critical error determining discriminator input dimension: {e}. Aborting.", exc_info=True)
        return

    # --- 4. Create Discriminator ---
    logger.info("\n--- Step 4: Creating Domain Discriminator ---")
    try:
        D = get_discriminator(
            name="default",
            input_dim=discriminator_input_dim,
            hidden_dim=1024
        ).to(device)
    except Exception as e:
        logger.error(f"Critical error creating discriminator: {e}. Aborting.", exc_info=True)
        return

    # --- 5. Run ADDA Training ---
    logger.info("\n--- Step 5: Starting ADDA Training Loop ---")
    if not adda_source_img_dir_for_loader.exists() or not any(adda_source_img_dir_for_loader.iterdir()) or \
       not adda_target_img_dir_for_loader.exists() or not any(adda_target_img_dir_for_loader.iterdir()):
        logger.error("ADDA source or target image directories for DataLoader are empty or missing. "
                     f"Source: {adda_source_img_dir_for_loader}, Target: {adda_target_img_dir_for_loader}. Aborting ADDA training.")
        return

    try:
        F_t_adapted = train_adda(
            F_s=F_s,
            F_t=F_t,
            D=D,
            adaptable_params_F_t=adaptable_params_F_t,
            hook_layer_indices_for_adda=adapt_and_hook_layer_indices,
            source_adda_img_dir=adda_source_img_dir_for_loader,
            target_adda_img_dir=adda_target_img_dir_for_loader,
            experiment_path=current_experiment_path,
            img_size=img_size_adda,
            batch_size=batch_size_adda,
            learning_rate_t=learning_rate_f_t,
            learning_rate_d=learning_rate_d,
            num_epochs=num_epochs_adda,
            device_str=device.type,
            save_intermediate_models=True
        )
    except ValueError as ve: # Catch specific errors like empty datasets
        logger.error(f"ValueError during ADDA training: {ve}. Aborting.", exc_info=True)
        return
    except Exception as e:
        logger.error(f"Critical error during ADDA training: {e}. Aborting.", exc_info=True)
        return

    # --- 6. Save Final Adapted Model ---
    logger.info("\n--- Step 6: Saving Final Adapted Ultralytics Model ---")
    final_adapted_model_path = current_experiment_path / f"{Path(pretrained_model_pt_path).stem}_ADDA_adapted.pt"
    try:
        save_adapted_ultralytics_model(
            original_model_wrapper=original_model_wrapper,
            adapted_F_t_state_dict=F_t_adapted.state_dict(),
            adapt_layer_indices=adapt_and_hook_layer_indices,
            output_pt_path=final_adapted_model_path,
            device_str=device.type
        )
        logger.info(f"Successfully saved final ADDA-adapted model to: {final_adapted_model_path}")
    except Exception as e:
        logger.error(f"Error saving final adapted model: {e}. Attempting to save F_t state_dict directly.", exc_info=True)
        fallback_save_path = current_experiment_path / f"{Path(pretrained_model_pt_path).stem}_ADDA_adapted_Ft_statedict_only.pt"
        torch.save(F_t_adapted.state_dict(), fallback_save_path)
        logger.info(f"Saved F_t_adapted state_dict directly to: {fallback_save_path}")

    logger.info(f"\nADDA process finished for experiment: {experiment_name}")
    logger.info(f"All outputs and logs are in: {current_experiment_path}")
    logger.info(f"Final adapted model (if successful): {final_adapted_model_path}")

