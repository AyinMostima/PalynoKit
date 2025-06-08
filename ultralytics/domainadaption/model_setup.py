import logging
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO, RTDETR
# Correctly import the specific model types if available, or use their string names if that's more robust
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel, ClassificationModel
# Attempt to import RTDETRDetectionModel, if it's not directly available, we might need to adjust the check
try:
    from ultralytics.nn.tasks import RTDETRDetectionModel
except ImportError:
    RTDETRDetectionModel = None # Fallback if not directly importable this way
    logger.warning("Could not directly import RTDETRDetectionModel from ultralytics.nn.tasks. Type checking for RTDETR might be less specific.")
import copy # Make sure copy is imported

from .utils import get_device # Assuming utils.py is in the same package

logger = logging.getLogger(__name__)



# Global dictionary to store features from hooks
# Key: (model_id, layer_idx), Value: feature_tensor
FEATURES_STORAGE = {}


def get_features_output_hook(module, input, output, model_id: str, layer_idx: int):
    """Hook to capture the output features of a specific module."""
    global FEATURES_STORAGE
    # For RTDETR, decoder output might be a tuple. We're usually interested in feature maps.
    # Typically, hooks are on Conv layers or blocks which output tensors.
    # If output is a tuple, try to get the first tensor element, or log a warning.
    feature_to_store = output
    if isinstance(output, tuple):
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            feature_to_store = output[0]
            # logger.debug(f"Hook on {model_id} layer {layer_idx}: Output is tuple, taking first element of shape {feature_to_store.shape}")
        else:
            logger.warning(
                f"Hook on {model_id} layer {layer_idx}: Output is tuple but first element is not a tensor or tuple is empty. Storing full tuple. Type: {type(output)}")

    FEATURES_STORAGE[(model_id, layer_idx)] = feature_to_store


def add_hooks_to_model(
        ultralytics_model_internal: nn.Module,  # This is model.model (the actual nn.Sequential or similar)
        model_id: str,
        layer_indices: list[int]
) -> list:
    """
    Adds forward hooks to specified layers of the model.model attribute.
    Args:
        ultralytics_model_internal: The core nn.Module (e.g., DetectionModel.model or RTDETRModel.model).
        model_id: A string identifier for the model (e.g., 'source' or 'target').
        layer_indices: A list of integer indices for layers in model.model to hook.
    Returns:
        A list of hook handles.
    """
    handles = []
    if not hasattr(ultralytics_model_internal, 'model') or not isinstance(ultralytics_model_internal.model,
                                                                          nn.Sequential):
        logger.error(f"Model structure for '{model_id}' is not as expected. "
                     f"Missing 'model' attribute or it's not nn.Sequential. "
                     f"Actual type: {type(ultralytics_model_internal.model if hasattr(ultralytics_model_internal, 'model') else ultralytics_model_internal)}")
        # Try to hook ultralytics_model_internal directly if it's a Sequential
        if isinstance(ultralytics_model_internal, nn.Sequential):
            logger.warning(f"Attempting to hook layers directly on '{model_id}' as it's an nn.Sequential.")
            model_to_hook = ultralytics_model_internal
        else:
            raise ValueError(f"Cannot attach hooks to {model_id}. Model structure incompatible.")
    else:
        model_to_hook = ultralytics_model_internal.model  # The nn.Sequential part

    for idx in layer_indices:
        try:
            layer = model_to_hook[idx]
            handle = layer.register_forward_hook(
                lambda module, input, output, model_id=model_id, layer_idx=idx: \
                    get_features_output_hook(module, input, output, model_id, layer_idx)
            )
            handles.append(handle)
            logger.info(f"Registered hook on model '{model_id}', layer {idx} ({layer.__class__.__name__})")
        except IndexError:
            logger.error(
                f"Error: Layer index {idx} out of range for model '{model_id}' (max index {len(model_to_hook) - 1}).")
            # Clean up already registered hooks before raising
            for h in handles: h.remove()
            raise
        except Exception as e:
            logger.error(f"Error registering hook on model '{model_id}', layer {idx}: {e}")
            for h in handles: h.remove()
            raise
    return handles


def remove_hooks_from_model(handles: list):
    """Removes all registered hooks."""
    for handle in handles:
        handle.remove()
    logger.info(f"Removed {len(handles)} hooks.")
    FEATURES_STORAGE.clear()  # Clear stored features


def load_ultralytics_model(model_path: str, device_str: str = "auto"):
    """Loads an Ultralytics model (.pt or .yaml) and returns the wrapper and core nn.Module."""
    device = get_device(device_str)
    model_path_obj = Path(model_path)

    SupportedModelClasses = [YOLO, RTDETR]
    if 'rtdetr' in model_path_obj.name.lower():
        SupportedModelClasses = [RTDETR, YOLO]

    loaded_model_wrapper = None
    core_nn_module = None

    # Define a tuple of known base model types for isinstance check
    # Add more Ultralytics base task model types here if needed
    known_task_models = (DetectionModel, SegmentationModel, PoseModel, ClassificationModel)
    if RTDETRDetectionModel:  # Only add if successfully imported
        known_task_models += (RTDETRDetectionModel,)
    else:  # If RTDETRDetectionModel couldn't be imported, rely on the wrapper type or broader checks
        logger.info("RTDETRDetectionModel not imported, will rely on wrapper type for RTDETR identification if needed.")

    for ModelClass in SupportedModelClasses:
        try:
            logger.info(f"Attempting to load model {model_path} as {ModelClass.__name__}.")
            model_wrapper_candidate = ModelClass(str(model_path_obj))  # Ensure model_path is string for constructor

            candidate_core_model = None
            if hasattr(model_wrapper_candidate, 'model') and model_wrapper_candidate.model is not None:
                # Check if model_wrapper_candidate.model is one of the known task model types
                if isinstance(model_wrapper_candidate.model, known_task_models):
                    candidate_core_model = model_wrapper_candidate.model
                else:
                    # Fallback: if it's not a known task model but is an nn.Module, use it.
                    # This might happen if the structure is simpler or custom.
                    if isinstance(model_wrapper_candidate.model, nn.Module):
                        logger.warning(
                            f"model_wrapper.model for {ModelClass.__name__} is an nn.Module but not a recognized Ultralytics task type. Using it as core model. Type: {type(model_wrapper_candidate.model)}")
                        candidate_core_model = model_wrapper_candidate.model
                    else:
                        logger.debug(
                            f"model_wrapper.model for {ModelClass.__name__} is not a recognized task model or nn.Module. Type: {type(model_wrapper_candidate.model)}. Skipping this load attempt.")
                        continue
            elif isinstance(model_wrapper_candidate, nn.Module) and not hasattr(model_wrapper_candidate, 'model'):
                # This case is less likely for .pt files from YOLO/RTDETR classes but good to consider
                logger.warning(
                    f"Model loaded with {ModelClass.__name__}, and the wrapper itself is the nn.Module (no '.model' attribute or it's None). "
                    "This might affect saving if a dedicated wrapper was expected for specific methods.")
                candidate_core_model = model_wrapper_candidate
            else:
                logger.debug(
                    f"Model loaded with {ModelClass.__name__} but could not identify a suitable core nn.Module. Skipping this load attempt.")
                continue

            if candidate_core_model is None:
                logger.debug(f"No core_nn_module identified for {ModelClass.__name__}. Skipping.")
                continue

            # Successfully identified a core model
            core_nn_module = candidate_core_model
            loaded_model_wrapper = model_wrapper_candidate

            # Store the original path and loading class for reloading later, critical for saving
            loaded_model_wrapper.original_ckpt_path = str(model_path_obj)
            loaded_model_wrapper.model_class_type = ModelClass

            core_nn_module.to(device)
            logger.info(
                f"Successfully loaded model {model_path} with {ModelClass.__name__} onto {device}. Core module type: {type(core_nn_module)}")
            break
        except Exception as e:
            logger.warning(
                f"Failed to load model {model_path} with {ModelClass.__name__}: {e}. Trying next model class if available.")
            # logger.debug(f"Exception details for {ModelClass.__name__}:", exc_info=True) # Uncomment for more debug info
            continue

    if loaded_model_wrapper is None or core_nn_module is None:
        logger.error(
            f"Failed to load model {model_path} with any of the supported classes ({[cls.__name__ for cls in SupportedModelClasses]}).")
        raise ValueError(f"Could not load model {model_path}")

    return loaded_model_wrapper, core_nn_module




def create_adda_feature_extractors(
        pretrained_model_path: str,
        adapt_layer_indices: list[int],
        device_str: str = "auto"
):
    """
    Creates the source (F_s) and target (F_t) feature extractors for ADDA.
    F_s is a frozen copy of the pretrained model.
    F_t is a copy where only specified layers are trainable.

    Returns:
        F_s (nn.Module): Frozen source feature extractor.
        F_t (nn.Module): Target feature extractor with adaptable layers.
        adaptable_params_F_t (list): List of parameters in F_t that are trainable.
        original_model_wrapper: The loaded Ultralytics model wrapper (for saving later).
    """
    device = get_device(device_str)

    logger.info(f"Creating ADDA feature extractors from: {pretrained_model_path}")
    logger.info(f"Layers to adapt in F_t: {adapt_layer_indices}")

    original_model_wrapper, core_model_nn_module = load_ultralytics_model(pretrained_model_path, device_str)

    # F_s: Frozen source model
    F_s = copy.deepcopy(core_model_nn_module).to(device)
    for param in F_s.parameters():
        param.requires_grad = False
    F_s.eval()
    logger.info("Created F_s (source feature extractor) and froze all parameters.")

    # F_t: Target model, initially frozen, then unfreeze adaptable layers
    F_t = copy.deepcopy(core_model_nn_module).to(device)
    for param in F_t.parameters():
        param.requires_grad = False

    adaptable_params_F_t = []
    # The actual layers are within F_t.model (which is an nn.Sequential)
    if not hasattr(F_t, 'model') or not isinstance(F_t.model, nn.Sequential):
        logger.error("F_t.model is not an nn.Sequential module. Cannot unfreeze layers by index.")
        # Fallback: if F_t itself is sequential (e.g. custom model not wrapped in DetectionModel)
        if isinstance(F_t, nn.Sequential):
            model_sequence_to_unfreeze = F_t
            logger.warning("F_t itself is nn.Sequential. Will attempt to unfreeze layers directly on F_t.")
        else:
            raise ValueError("F_t.model is not nn.Sequential. Cannot proceed with unfreezing layers.")
    else:
        model_sequence_to_unfreeze = F_t.model

    for idx in adapt_layer_indices:
        try:
            layer_to_adapt = model_sequence_to_unfreeze[idx]
            for param in layer_to_adapt.parameters():
                param.requires_grad = True
                adaptable_params_F_t.append(param)
            logger.info(f"  Unfroze parameters for layer {idx} ({layer_to_adapt.__class__.__name__}) in F_t.")
        except IndexError:
            logger.error(
                f"  Error: Layer index {idx} out of range for F_t.model (max index {len(model_sequence_to_unfreeze) - 1}).")
            raise
        except Exception as e:
            logger.error(f"  Error processing layer {idx} in F_t for unfreezing: {e}")
            raise

    if not adaptable_params_F_t:
        logger.warning("No parameters were unfrozen in F_t. Check adapt_layer_indices or model structure.")
    else:
        logger.info(f"Total {len(adaptable_params_F_t)} parameter groups made adaptable in F_t.")

    F_t.train()  # Set F_t to train mode (important for BatchNorm, Dropout in adaptable layers)

    return F_s, F_t, adaptable_params_F_t, original_model_wrapper


def determine_feature_dim_for_discriminator(
        model_for_dim_check: nn.Module,  # This should be F_s or F_t (the core nn.Module)
        hook_target_layer_indices: list[int],
        img_size: int,
        device_str: str = "auto"
) -> int:
    """
    Performs a dummy forward pass to determine the combined feature dimension
    from the outputs of the hooked layers. This dimension will be the input
    size for the discriminator.
    Assumes hooks will populate FEATURES_STORAGE.
    The discriminator will use AdaptiveAvgPool2d, so we sum channel dimensions.
    """
    device = get_device(device_str)
    logger.info("Determining feature dimension for discriminator...")

    model_for_dim_check.to(device).eval()  # Ensure it's on device and in eval mode for this check

    # Clear storage and add temporary hooks
    FEATURES_STORAGE.clear()
    temp_hook_handles = add_hooks_to_model(model_for_dim_check, model_id='dim_check',
                                           layer_indices=hook_target_layer_indices)

    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        _ = model_for_dim_check(dummy_input)  # Forward pass to trigger hooks

    combined_feature_dim = 0
    for idx in hook_target_layer_indices:
        feature_key = ('dim_check', idx)
        if feature_key in FEATURES_STORAGE:
            feat = FEATURES_STORAGE[feature_key]
            if isinstance(feat, torch.Tensor):
                # Regardless of H, W, AdaptiveAvgPool2d in discriminator will reduce to (B, C, 1, 1)
                # So, we sum the channel dimensions (dim 1)
                if feat.ndim >= 2:  # Should be B, C, ... or B, N, D
                    # If B,C,H,W -> feat.shape[1] is C
                    # If B,N,D (transformer output) -> feat.shape[-1] is D (embedding_dim) if we treat N as spatial.
                    # The hook logic tries to get a primary tensor.
                    # For typical Conv layers, shape[1] is channels.
                    # For Transformer blocks, output might be [B, NumTokens, EmbedDim].
                    # If hook stores [B,N,D], then D (feat.shape[2]) is the channel-like dimension.
                    # Let's assume features are [B, C, ...] or [B, ..., C]
                    # The current hook captures the raw output.
                    # If feat.ndim == 4 (B,C,H,W): channels = feat.shape[1]
                    # If feat.ndim == 3 (B,N,D e.g. from some transformer blocks): channels = feat.shape[2]
                    # If feat.ndim == 2 (B, FlatFeatures): channels = feat.shape[1] (less common from intermediate layers)

                    current_channels = 0
                    if feat.ndim == 4:  # B, C, H, W
                        current_channels = feat.shape[1]
                    elif feat.ndim == 3:  # B, N, D (Tokens, EmbeddingDim) or B, C, FlatSpatial
                        current_channels = feat.shape[-1]  # Assume last dim is channel/embedding like
                        # OR if it's B, C, FlatSpatial, then feat.shape[1] is C
                        # This part needs care depending on what kind of layers are hooked.
                        # For safety, and common CNNs, let's prioritize shape[1] if > 1, else shape[-1].
                        # The discriminator will apply pooling. The key is the number of channels.
                        # DefaultDomainDiscriminator handles 3D input by mean over last spatial dim, then concat channels.
                        # So, for 3D [B, C, N_spatial], it takes C. For [B, N_spatial, D_channels] it takes D.
                        # Let's assume it's [Batch, Channels, *SpatialDims] or [Batch, *SpatialDims, Channels]
                        # The discriminator's pooling logic clarifies this:
                        # - 4D [B,C,H,W] -> pool -> [B,C,1,1] -> flatten -> [B,C]
                        # - 3D [B,C,N] -> mean(dim=2) -> [B,C]
                        # So, feat.shape[1] is the channel dimension for 3D if it's [B,C,N]
                        # If it's [B,N,D] (num_tokens, embed_dim), then D is channel-like, so feat.shape[2]
                        # Given the ambiguity, for now, let's assume for CNNs it's feat.shape[1]
                        # If a transformer block (outputting B,N,D) is hooked, this might need adjustment
                        # or the hook needs to be on a layer that reshapes to B,C,H,W before this.
                        # For RT-DETR, encoder outputs are typically [B, H*W, C]. Decoder inputs are [B,C,H,W].
                        # If we hook encoder layers, it's [B, Tokens, Channels]. Channel dim is feat.shape[2].
                        # If we hook backbone/neck CNN layers, it's [B, Channels, H, W]. Channel dim is feat.shape[1].

                        if feat.shape[1] > 1 and feat.ndim > 2:  # Likely B,C,H,W or B,C,N
                            current_channels = feat.shape[1]
                        else:  # Potentially B,N,D or B,Flat
                            current_channels = feat.shape[-1]

                    elif feat.ndim == 2:  # B, C (already pooled or flattened)
                        current_channels = feat.shape[1]
                    else:
                        logger.warning(
                            f"Feature from layer {idx} has unexpected ndim: {feat.ndim}. Shape: {feat.shape}. Cannot determine channels.")
                        continue

                    logger.info(
                        f"  Hooked feature from layer {idx} ('dim_check') shape: {feat.shape}. Deduced channels: {current_channels}")
                    combined_feature_dim += current_channels
                else:
                    logger.warning(f"  Hook for layer {idx} ('dim_check') captured non-Tensor: {type(feat)}")
            else:
                logger.error(f"  Hook failed to capture feature from layer {idx} ('dim_check').")
                remove_hooks_from_model(temp_hook_handles)
                raise RuntimeError(f"Hook failed for layer {idx} during dimension check.")

    remove_hooks_from_model(temp_hook_handles)  # Clean up temp hooks

    if combined_feature_dim == 0:
        logger.error("Failed to determine feature dimension (result is 0). Check hook indices and model structure.")
        raise ValueError("Feature dimension for discriminator is 0.")

    logger.info(f"Determined combined feature dimension for discriminator input: {combined_feature_dim}")
    return combined_feature_dim


def save_adapted_ultralytics_model(
        original_model_wrapper,  # The wrapper loaded initially (YOLO or RTDETR instance)
        adapted_F_t_state_dict: dict,
        adapt_layer_indices: list[int],
        output_pt_path: Path,
        device_str: str = "auto"
):
    """
    Injects the adapted weights from F_t into a fresh instance of the original model
    (reloaded from its original path) and saves it.
    """
    device = get_device(device_str)
    logger.info(f"Preparing to save adapted model to: {output_pt_path}")

    if not hasattr(original_model_wrapper, 'original_ckpt_path') or not original_model_wrapper.original_ckpt_path:
        logger.error(
            "Original model wrapper does not have 'original_ckpt_path' attribute. Cannot reliably reload for saving.")
        # Fallback: Try to use the current original_model_wrapper structure if path is missing.
        # This might lead to issues if its state is not pristine.
        logger.warning("Attempting to use the existing original_model_wrapper instance for saving. This is a fallback.")
        new_model_wrapper = copy.deepcopy(original_model_wrapper)
        # We need to ensure its weights are the *original* ones, not potentially modified ones.
        # This fallback is risky. Best to ensure original_ckpt_path is set.
        # For now, if this happens, we proceed but with a high risk of incorrect base for adapted model.
        # A safer fallback if original_ckpt_path is missing would be to error out or only save state_dict.
        # However, the load_ultralytics_model should now set this.
        if not (hasattr(original_model_wrapper, 'original_ckpt_path') and original_model_wrapper.original_ckpt_path):
            logger.error(
                "CRITICAL: Fallback to deepcopy failed or original_ckpt_path missing. Cannot proceed with safe model saving.")
            raise ValueError("original_ckpt_path is missing on the model wrapper, cannot save adapted model correctly.")

    else:
        logger.info(
            f"Re-loading original model structure and weights from: {original_model_wrapper.original_ckpt_path}")
        # Determine which class to use for reloading based on what successfully loaded it initially
        ModelLoadClass = YOLO  # Default
        if hasattr(original_model_wrapper, 'model_class_type'):
            ModelLoadClass = original_model_wrapper.model_class_type
            logger.info(f"Using {ModelLoadClass.__name__} to reload the original model.")
        else:
            logger.warning(
                "model_class_type not found on original_model_wrapper. Defaulting to YOLO for reload. This might fail if original was RTDETR.")

        try:
            # Reload a fresh instance from the original .pt file path
            new_model_wrapper = ModelLoadClass(original_model_wrapper.original_ckpt_path)
        except Exception as e_reload:
            logger.error(
                f"Failed to reload original model from {original_model_wrapper.original_ckpt_path} using {ModelLoadClass.__name__}: {e_reload}")
            logger.error("Cannot proceed with saving the adapted model in Ultralytics format.")
            raise

    final_model_nn_module = new_model_wrapper.model
    final_model_nn_module.to(device)  # Ensure it's on the correct device

    current_state_dict = final_model_nn_module.state_dict()  # Get state_dict of the reloaded (original) model

    updated_count = 0
    skipped_count = 0

    target_key_prefixes = tuple(f'model.{i}.' for i in adapt_layer_indices)

    for key_from_adapted_f_t, adapted_param_value in adapted_F_t_state_dict.items():
        is_adaptable_layer_param = any(key_from_adapted_f_t.startswith(prefix) for prefix in target_key_prefixes)

        if is_adaptable_layer_param:
            if key_from_adapted_f_t in current_state_dict:
                if current_state_dict[key_from_adapted_f_t].shape == adapted_param_value.shape:
                    current_state_dict[key_from_adapted_f_t].copy_(adapted_param_value.to(
                        current_state_dict[key_from_adapted_f_t].device))  # Ensure tensor devices match
                    updated_count += 1
                else:
                    logger.warning(f"  Shape mismatch for key '{key_from_adapted_f_t}': "
                                   f"current model {current_state_dict[key_from_adapted_f_t].shape}, "
                                   f"adapted {adapted_param_value.shape}. Skipped.")
                    skipped_count += 1
            else:
                logger.warning(
                    f"  Key '{key_from_adapted_f_t}' from adapted F_t not found in the reloaded model instance. Skipped.")
                skipped_count += 1

    if updated_count == 0 and adapt_layer_indices:  # only warn if we expected updates
        logger.warning("No parameters from adapted layers were updated during state_dict injection. "
                       "Check adapt_layer_indices and key matching (e.g., 'model.LAYER_IDX.').")
    elif not adapt_layer_indices:
        logger.info(
            "No adapt_layer_indices provided, so no specific layer weights were injected from F_t's state_dict based on indices.")

    logger.info(f"Weight injection complete: {updated_count} parameter tensors updated, {skipped_count} skipped.")

    try:
        final_model_nn_module.load_state_dict(current_state_dict)
        logger.info("Updated state_dict successfully loaded into the reloaded model instance's nn.Module.")
    except RuntimeError as e:
        logger.error(f"Error loading updated state_dict into new model instance: {e}", exc_info=True)
        raise

    try:
        new_model_wrapper.model = final_model_nn_module
        # new_model_wrapper.trainer = None # Often good to clear this
        # new_model_wrapper.ckpt = final_model_nn_module.state_dict() # Update ckpt in wrapper if save method uses it

        output_pt_path.parent.mkdir(parents=True, exist_ok=True)
        new_model_wrapper.save(str(output_pt_path))  # This method handles .pt file creation
        logger.info(f"Successfully saved adapted model using Ultralytics wrapper to: {output_pt_path}")
    except Exception as e:
        logger.error(f"Error saving final adapted model using Ultralytics wrapper: {e}", exc_info=True)
        # Fallback as before
        fallback_path = output_pt_path.with_name(output_pt_path.stem + "_statedict_fallback.pt")
        torch.save(final_model_nn_module.state_dict(), fallback_path)
        logger.info(f"Ultralytics wrapper save failed. Saved only state_dict to: {fallback_path}")
        raise


if __name__ == '__main__':
    from .utils import setup_logging

    setup_logging()

    logger.info("Testing model_setup.py...")
    # Define a dummy .pt model path (e.g., yolov8n.pt or your RT-DETR model)
    # IMPORTANT: Replace with a valid path to a .pt model on your system for testing
    TEST_MODEL_PT_PATH = "yolov8n.pt"  # or "your_rtdetr_model.pt"
    if not Path(TEST_MODEL_PT_PATH).exists():
        logger.error(f"Test model {TEST_MODEL_PT_PATH} not found. Skipping model_setup tests.")
    else:
        ADAPT_LAYERS = [4, 6]  # Example layers to adapt (these are early layers in YOLOv8n backbone)

        # 1. Test create_adda_feature_extractors
        logger.info("\n--- Testing create_adda_feature_extractors ---")
        try:
            F_s, F_t, adaptable_params, orig_wrapper = create_adda_feature_extractors(
                pretrained_model_path=TEST_MODEL_PT_PATH,
                adapt_layer_indices=ADAPT_LAYERS,
                device_str="cpu"
            )
            logger.info(f"F_s type: {type(F_s)}, F_t type: {type(F_t)}")
            logger.info(f"Number of adaptable parameter groups in F_t: {len(adaptable_params)}")

            # Check if F_s is frozen
            fs_frozen = all(not p.requires_grad for p in F_s.parameters())
            logger.info(f"F_s parameters frozen: {fs_frozen}")
            assert fs_frozen

            # Check if F_t adaptable layers are unfrozen
            # This is harder to check directly without knowing param names, but adaptable_params should exist
            assert len(adaptable_params) > 0 if ADAPT_LAYERS else True

            # 2. Test determine_feature_dim_for_discriminator
            logger.info("\n--- Testing determine_feature_dim_for_discriminator ---")
            # Use F_s (or F_t) for dimension check. It's the core nn.Module.
            # The layers to hook for discriminator input might be different from ADAPT_LAYERS.
            # For this test, let's use the same ADAPT_LAYERS as hook targets.
            HOOK_LAYERS_FOR_DISC = ADAPT_LAYERS  # Or choose other layers like [10, 12, 14] from your example
            if not HOOK_LAYERS_FOR_DISC:  # Ensure there's something to hook
                HOOK_LAYERS_FOR_DISC = [2, 3]  # Default if ADAPT_LAYERS was empty for test

            try:
                # F_s is DetectionModel or RTDETRModel. Pass this.
                # add_hooks_to_model expects the .model attribute (the nn.Sequential) IF F_s is DetectionModel etc.
                # However, determine_feature_dim expects the module that can be called: F_s(dummy_input)
                # So, pass F_s itself. The add_hooks_to_model will handle F_s.model.
                feature_dim = determine_feature_dim_for_discriminator(
                    model_for_dim_check=F_s,  # Pass the DetectionModel/RTDETRModel instance
                    hook_target_layer_indices=HOOK_LAYERS_FOR_DISC,
                    img_size=320,  # Smaller size for faster test
                    device_str="cpu"
                )
                logger.info(f"Determined discriminator input feature dimension: {feature_dim}")
                assert feature_dim > 0
            except Exception as e_dim:
                logger.error(f"Error in determine_feature_dim: {e_dim}", exc_info=True)

            # 3. Test save_adapted_ultralytics_model (simulated)
            logger.info("\n--- Testing save_adapted_ultralytics_model (simulated) ---")
            # Simulate that F_t has been trained, so its state_dict is different.
            # For this test, we'll just use F_t's current state_dict.
            # In a real scenario, this would be F_t.state_dict() AFTER ADDA training.
            adapted_f_t_state_dict = F_t.state_dict()

            # Create a dummy output path
            temp_output_dir = Path("./temp_adda_output_model_setup_test")
            temp_output_dir.mkdir(exist_ok=True)
            output_adapted_model_path = temp_output_dir / "adapted_test_model.pt"

            try:
                save_adapted_ultralytics_model(
                    original_model_wrapper=orig_wrapper,  # Pass the original wrapper
                    adapted_F_t_state_dict=adapted_f_t_state_dict,
                    adapt_layer_indices=ADAPT_LAYERS,
                    output_pt_path=output_adapted_model_path,
                    device_str="cpu"
                )
                logger.info(f"Simulated saving successful. Check: {output_adapted_model_path}")
                assert output_adapted_model_path.exists()
            except Exception as e_save:
                logger.error(f"Error in save_adapted_ultralytics_model: {e_save}", exc_info=True)
            finally:
                # Clean up dummy file and dir
                if output_adapted_model_path.exists(): output_adapted_model_path.unlink()
                if temp_output_dir.exists(): shutil.rmtree(temp_output_dir)

        except Exception as e:
            logger.error(f"Error during model_setup tests: {e}", exc_info=True)