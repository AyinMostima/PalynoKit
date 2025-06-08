
import logging
import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data_preparation import AddaDomainDataset  # Assuming AddaDomainDataset is correctly defined
from .model_setup import add_hooks_to_model, remove_hooks_from_model, FEATURES_STORAGE
from .utils import get_device, SOURCE_DOMAIN_LABEL, TARGET_DOMAIN_LABEL  # Assuming these are defined

logger = logging.getLogger(__name__)


def train_adda(
        F_s: nn.Module,
        F_t: nn.Module,
        D: nn.Module,
        adaptable_params_F_t: list,
        hook_layer_indices_for_adda: list[int],  # <<< CORRECTLY ADDED ARGUMENT
        source_adda_img_dir: Path,
        target_adda_img_dir: Path,
        experiment_path: Path,
        img_size: int = 640,
        batch_size: int = 4,
        learning_rate_t: float = 1e-5,
        learning_rate_d: float = 1e-4,
        num_epochs: int = 50,
        device_str: str = "auto",
        save_intermediate_models: bool = False
):
    """
    Performs the Adversarial Discriminative Domain Adaptation (ADDA) training.
    """
    device = get_device(device_str)
    logger.info("Starting ADDA training process...")
    # ensure_log_dir_exists(experiment_path) # This function was defined in adda_trainer, ensure it's available or handled by experiment_path creation

    F_s.to(device).eval()
    F_t.to(device)
    D.to(device).train()

    logger.info("Setting up ADDA DataLoaders...")
    source_domain_dataset = AddaDomainDataset(source_adda_img_dir, img_size, SOURCE_DOMAIN_LABEL)
    target_domain_dataset = AddaDomainDataset(target_adda_img_dir, img_size, TARGET_DOMAIN_LABEL)

    if not source_domain_dataset or not target_domain_dataset or \
            len(source_domain_dataset) == 0 or len(target_domain_dataset) == 0:
        logger.error("Source or Target ADDA domain dataset is empty. Cannot train.")
        raise ValueError("ADDA domain dataset(s) empty.")

    source_loader = DataLoader(
        source_domain_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )
    target_loader = DataLoader(
        target_domain_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )
    logger.info(f"Source domain ADDA dataset size: {len(source_domain_dataset)}, loader batches: {len(source_loader)}")
    logger.info(f"Target domain ADDA dataset size: {len(target_domain_dataset)}, loader batches: {len(target_loader)}")

    optimizer_t = None
    if adaptable_params_F_t:
        optimizer_t = optim.Adam(adaptable_params_F_t, lr=learning_rate_t, betas=(0.5, 0.999))
        logger.info(
            f"Optimizer for F_t adaptable layers (Adam, lr={learning_rate_t}) managing {sum(p.numel() for p in adaptable_params_F_t)} parameters.")
    else:
        logger.warning("No adaptable parameters found for F_t. Adversarial training for F_t will not occur.")

    optimizer_d = optim.Adam(D.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))
    logger.info(
        f"Optimizer for Discriminator D (Adam, lr={learning_rate_d}) managing {sum(p.numel() for p in D.parameters())} parameters.")

    criterion_d = nn.BCEWithLogitsLoss()

    global FEATURES_STORAGE
    hook_handles_s = []
    hook_handles_t = []

    results_log = []
    num_batches = min(len(source_loader), len(target_loader))
    if num_batches == 0:
        logger.error("Zero batches for training. DataLoaders might be empty. Aborting.")
        return F_t

    logger.info(f"Starting ADDA training for {num_epochs} epochs, {num_batches} batches per epoch.")

    if not hook_layer_indices_for_adda:
        logger.warning(
            "'hook_layer_indices_for_adda' is empty. No hooks will be attached. Discriminator will not get features. Training might be ineffective.")
    else:
        try:
            logger.info(f"Attaching hooks to F_s and F_t for layers: {hook_layer_indices_for_adda}")
            hook_handles_s = add_hooks_to_model(F_s, 'source_hooked', hook_layer_indices_for_adda)
            hook_handles_t = add_hooks_to_model(F_t, 'target_hooked', hook_layer_indices_for_adda)
            if len(hook_handles_s) != len(hook_layer_indices_for_adda) or \
                    len(hook_handles_t) != len(hook_layer_indices_for_adda):
                logger.error("Failed to attach all necessary hooks. Aborting training.")
                remove_hooks_from_model(hook_handles_s + hook_handles_t)
                return F_t
        except Exception as e_hook:
            logger.error(f"Error attaching hooks: {e_hook}. Aborting training.", exc_info=True)
            remove_hooks_from_model(hook_handles_s + hook_handles_t)
            return F_t

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss_d = 0.0
        running_loss_adv = 0.0

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for batch_idx in range(num_batches):
            try:
                source_imgs, source_domain_labels_actual = next(source_iter)
                target_imgs, target_domain_labels_actual = next(target_iter)
            except StopIteration:
                logger.warning("DataLoader iterator exhausted prematurely. Breaking batch loop.")
                break

            source_imgs = source_imgs.to(device)
            target_imgs = target_imgs.to(device)
            labels_source_real = source_domain_labels_actual.to(device)
            labels_target_real = target_domain_labels_actual.to(device)
            labels_target_fake_to_source = torch.full_like(labels_target_real, SOURCE_DOMAIN_LABEL).to(device)

            FEATURES_STORAGE.clear()

            # --- 1. Train Discriminator D ---
            D.train()
            if optimizer_t:
                for param in adaptable_params_F_t: param.requires_grad_(False)

            optimizer_d.zero_grad()

            F_s.eval()  # Ensure F_s is in eval mode
            with torch.no_grad():
                _ = F_s(source_imgs)

            F_t.eval()
            with torch.no_grad():
                _ = F_t(target_imgs)
            # F_t.train() # Set F_t back to train for its own update step later. Needs to be inside the optimizer_t block.

            source_features_list = []
            target_features_list = []
            if hook_layer_indices_for_adda:  # Only try to get features if hooks were intended
                source_features_list = [FEATURES_STORAGE.get(('source_hooked', idx)) for idx in
                                        hook_layer_indices_for_adda]
                target_features_list = [FEATURES_STORAGE.get(('target_hooked', idx)) for idx in
                                        hook_layer_indices_for_adda]

            if hook_layer_indices_for_adda and (
                    any(f is None for f in source_features_list) or any(f is None for f in target_features_list)):
                logger.warning(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}: Missing hooked features for D training. Skipping batch.")
                FEATURES_STORAGE.clear()
                if optimizer_t:
                    for param in adaptable_params_F_t: param.requires_grad_(True)
                continue

            if not hook_layer_indices_for_adda and not (
                    source_features_list and target_features_list):  # if no hooks, D can't get input unless it's designed differently
                logger.warning(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}: No hook indices provided, D cannot get features. Skipping D update.")
                loss_d = torch.tensor(0.0)  # No D loss if no features
            else:
                pred_s = D(source_features_list)
                pred_t = D(target_features_list)
                loss_d_s = criterion_d(pred_s, labels_source_real)
                loss_d_t = criterion_d(pred_t, labels_target_real)
                loss_d = (loss_d_s + loss_d_t) / 2.0
                loss_d.backward()
                optimizer_d.step()

            running_loss_d += loss_d.item()

            # --- 2. Train Target Feature Extractor F_t ---
            loss_adv = torch.tensor(0.0)  # Initialize
            if optimizer_t:
                F_t.train()  # Set F_t to train mode for its own update
                D.eval()
                for param in adaptable_params_F_t: param.requires_grad_(True)

                optimizer_t.zero_grad()
                FEATURES_STORAGE.clear()

                _ = F_t(target_imgs)

                target_features_adv_list = []
                if hook_layer_indices_for_adda:
                    target_features_adv_list = [FEATURES_STORAGE.get(('target_hooked', idx)) for idx in
                                                hook_layer_indices_for_adda]

                if hook_layer_indices_for_adda and any(f is None for f in target_features_adv_list):
                    logger.warning(
                        f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}: Missing hooked features for F_t training. Skipping F_t update.")
                    FEATURES_STORAGE.clear()
                    continue

                if not hook_layer_indices_for_adda and not target_features_adv_list:
                    logger.warning(
                        f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}: No hook indices, F_t cannot fool D. Skipping F_t update.")
                else:
                    pred_t_adv = D(target_features_adv_list)
                    loss_adv = criterion_d(pred_t_adv, labels_target_fake_to_source)
                    loss_adv.backward()
                    optimizer_t.step()

            running_loss_adv += loss_adv.item()

            if (batch_idx + 1) % (num_batches // 5 if num_batches >= 5 else 1) == 0:
                logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{num_batches}], "
                            f"Loss D: {loss_d.item():.4f}, Loss Adv (F_t): {loss_adv.item():.4f}")

        epoch_duration = time.time() - epoch_start_time
        avg_loss_d = running_loss_d / num_batches if num_batches > 0 else 0
        avg_loss_adv = running_loss_adv / num_batches if num_batches > 0 and optimizer_t else 0

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration:.2f}s. "
                    f"Avg Loss D: {avg_loss_d:.4f}, Avg Loss Adv (F_t): {avg_loss_adv:.4f}")

        results_log.append({
            'Epoch': epoch + 1,
            'Time_s': epoch_duration,
            'Loss_D': avg_loss_d,
            'Loss_Adv_Ft': avg_loss_adv
        })

        if save_intermediate_models and (epoch + 1) % (num_epochs // 10 if num_epochs >= 10 else 1) == 0 and (
                epoch + 1) > 0:
            intermediate_model_dir = experiment_path / "intermediate_models"
            intermediate_model_dir.mkdir(parents=True, exist_ok=True)
            intermediate_model_path = intermediate_model_dir / f"F_t_adapted_epoch_{epoch + 1}.pt"
            torch.save(F_t.state_dict(), intermediate_model_path)
            logger.info(f"Saved intermediate F_t state_dict to {intermediate_model_path}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logger.info("ADDA adaptation training finished!")

    logger.info("Cleaning up hooks...")
    remove_hooks_from_model(hook_handles_s)
    remove_hooks_from_model(hook_handles_t)
    logger.info("All hooks removed.")

    log_df = pd.DataFrame(results_log)
    log_dir = experiment_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = log_dir / "adda_training_log.csv"
    log_df.to_csv(log_csv_path, index=False)
    logger.info(f"Training log saved to: {log_csv_path}")

    return F_t
