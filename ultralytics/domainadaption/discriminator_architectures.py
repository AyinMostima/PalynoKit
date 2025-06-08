
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DefaultDomainDiscriminator(nn.Module):
    """
    A default domain discriminator network.
    Uses LayerNorm instead of BatchNorm for robustness to small batch sizes.
    """

    def __init__(self, input_total_channels: int, hidden_dim: int = 1024):
        super().__init__()
        self.input_total_channels = input_total_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # For pooling spatial dimensions of features

        self.model = nn.Sequential(
            nn.Linear(input_total_channels, hidden_dim),
            # nn.BatchNorm1d(hidden_dim), # Replaced
            nn.LayerNorm(hidden_dim),  # Replacement: Normalizes over the features of each sample
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.BatchNorm1d(hidden_dim // 2), # Replaced
            nn.LayerNorm(hidden_dim // 2),  # Replacement
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)  # Output a single logit for domain classification
        )
        logger.info(
            f"Initialized DefaultDomainDiscriminator with input_total_channels: {input_total_channels}, hidden_dim: {hidden_dim}. Using LayerNorm.")

    def forward(self, features_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the discriminator.
        Args:
            features_list: A list of feature tensors, one from each hooked layer.
        Returns:
            A tensor of logits [Batch, 1].
        """
        pooled_features = []
        for feat_map in features_list:
            if isinstance(feat_map, torch.Tensor):
                if feat_map.ndim == 2:
                    # This case assumes pre-flattened features for a single source, matching total channels
                    if feat_map.shape[1] == self.input_total_channels and len(features_list) == 1:
                        # If only one feature tensor is passed and it's already 2D [B, TotalChannels]
                        return self.model(feat_map)
                    else:
                        # If multiple 2D features or mismatch, this logic is insufficient without knowing individual expected channels
                        logger.error(
                            f"Cannot process 2D feature map of shape {feat_map.shape} in a multi-feature setup or if channels don't match input_total_channels correctly.")
                        raise ValueError(
                            f"Unsupported 2D feature map shape {feat_map.shape} for discriminator's current pooling logic.")
                elif feat_map.ndim == 3:  # B, C, N (e.g., B, Channels, NumTokens)
                    pooled = torch.mean(feat_map, dim=2)  # Average over the "spatial" or token dimension -> [B, C]
                    pooled_features.append(pooled)
                elif feat_map.ndim == 4:  # B, C, H, W (typical CNN output)
                    pooled = self.pool(feat_map)  # Output shape: [B, C, 1, 1]
                    pooled_features.append(torch.flatten(pooled, 1))  # Output shape: [B, C]
                else:
                    logger.warning(
                        f"Skipping feature map with unexpected ndim: {feat_map.ndim}, shape: {feat_map.shape}")
                    continue
            else:
                logger.warning(f"Skipping non-Tensor input in features_list: {type(feat_map)}")
                continue

        if not pooled_features:
            # This can happen if hook_layer_indices_for_adda is empty or all features are invalid
            logger.error("No valid features to process in discriminator after pooling/selection.")
            # Depending on batch_size, you might want to return a tensor of zeros
            # but raising an error is safer to indicate a problem upstream.
            raise ValueError("Discriminator received no valid features to process.")

        try:
            x = torch.cat(pooled_features, dim=1)  # Output shape: [B, total_channels]
        except Exception as e:
            logger.error(f"Error concatenating pooled features: {e}")
            for i, pf in enumerate(pooled_features):
                logger.error(f"Pooled feature {i} shape: {pf.shape}")
            raise

        if x.shape[1] != self.input_total_channels:
            logger.error(
                f"Discriminator input dimension mismatch! Expected {self.input_total_channels}, got {x.shape[1]}. "
                "This usually indicates an issue with hook setup or feature dimension calculation.")
            raise ValueError(
                f"Discriminator input dim mismatch: expected {self.input_total_channels}, got {x.shape[1]}")

        return self.model(x)


class SimpleDomainDiscriminator(nn.Module):
    """
    A simpler, potentially "weaker" domain discriminator.
    Fewer layers and smaller hidden dimension compared to the default.
    """

    def __init__(self, input_total_channels: int, hidden_dim: int = 256):  # Smaller hidden_dim
        super().__init__()
        self.input_total_channels = input_total_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.model = nn.Sequential(
            nn.Linear(input_total_channels, hidden_dim),
            # nn.BatchNorm1d(hidden_dim), # Consider removing BatchNorm for a "weaker" version
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5), # Consider reducing or removing Dropout
            nn.Linear(hidden_dim, 1)  # Directly to output or one more smaller layer
        )
        logger.info(
            f"Initialized SimpleDomainDiscriminator with input_total_channels: {input_total_channels}, hidden_dim: {hidden_dim}")

    def forward(self, features_list: list[torch.Tensor]) -> torch.Tensor:
        pooled_features = []
        for feat_map in features_list:
            if isinstance(feat_map, torch.Tensor):
                if feat_map.ndim == 4:  # B, C, H, W
                    pooled = self.pool(feat_map)
                    pooled_features.append(torch.flatten(pooled, 1))
                elif feat_map.ndim == 3:  # B, C, N or B, N, D
                    # Assuming channels are feat_map.shape[1] if B,C,N
                    # Or feat_map.shape[-1] if B,N,D
                    # For simplicity, let's assume mean over the last spatial dim as before
                    if feat_map.shape[1] > 1 and feat_map.shape[-1] > 1:  # ambiguous, use a heuristic
                        # if C is likely smaller than N for B,C,N or D is smaller than N for B,N,D
                        # This part should align with how features are determined.
                        # For now, stick to previous logic:
                        if feat_map.shape[1] < feat_map.shape[-1]:  # Likely B,C,N -> mean over N (dim 2)
                            pooled = torch.mean(feat_map, dim=2)
                        else:  # Likely B,N,D -> mean over N (dim 1)
                            pooled = torch.mean(feat_map, dim=1)  # this is probably wrong, D is channels.
                            # Correct for B,N,D: features are [B, Tokens, Channels], mean over Tokens
                            pooled = torch.mean(feat_map, dim=1)  # D becomes channels
                            # If features are B, C, Spatial_Flat, mean over Spatial_Flat (dim 2)
                            # pooled = torch.mean(feat_map, dim=2) # C becomes channels
                        # Let's simplify for this "weak" discriminator assuming features are pooled appropriately before concat
                        # or that the first dim after batch is channel-like after pooling
                        # The AdaptiveAvgPool2d handles 4D. For 3D, this pooling logic is critical.
                        # For simplicity let's assume the input list is already [B,C] per feature
                        # OR that the logic from DefaultDomainDiscriminator is fine.
                        # Sticking to previous logic for now:
                        current_channels_dim = -1
                        if feat_map.shape[1] > 1 and feat_map.ndim > 2:
                            current_channels_dim = 1  # B,C,H,W or B,C,N
                        else:
                            current_channels_dim = -1  # B,N,D or B,Flat -> last dim is channel

                        if current_channels_dim == 1:  # B,C,N
                            pooled = torch.mean(feat_map, dim=2)  # Result B,C
                        else:  # B,N,D
                            pooled = torch.mean(feat_map, dim=1)  # Result B,D (D treated as C)
                        pooled_features.append(pooled)


                elif feat_map.ndim == 2:  # B, C (already pooled/flattened)
                    pooled_features.append(feat_map)
                else:
                    logger.warning(
                        f"SimpleD: Skipping feature map with unexpected ndim: {feat_map.ndim}, shape: {feat_map.shape}")
                    continue
            else:
                logger.warning(f"SimpleD: Skipping non-Tensor input in features_list: {type(feat_map)}")
                continue

        if not pooled_features:
            logger.error("SimpleD: No valid features to process.")
            raise ValueError("SimpleDomainDiscriminator received no valid features.")

        try:
            x = torch.cat(pooled_features, dim=1)
        except Exception as e:
            logger.error(f"SimpleD: Error concatenating pooled features: {e}")
            raise

        if x.shape[1] != self.input_total_channels:
            logger.error(f"SimpleD: Input dimension mismatch! Expected {self.input_total_channels}, got {x.shape[1]}.")
            raise ValueError(f"SimpleD input dim mismatch: expected {self.input_total_channels}, got {x.shape[1]}")

        return self.model(x)

# You can add the new class definition here:
# For example, after the SimpleDomainDiscriminator class.

class SimplestDomainDiscriminator(nn.Module):
    """
    The simplest possible domain discriminator.
    It processes features similar to other discriminators but uses a single linear layer as its model.
    """
    def __init__(self, input_total_channels: int):
        super().__init__()
        self.input_total_channels = input_total_channels

        # Pooling layer, consistent with other discriminators for 4D feature maps
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # The model is a single linear layer mapping concatenated features to a single logit
        self.model = nn.Linear(input_total_channels, 1)

        # Logger info (optional, but consistent with your existing code)
        # Ensure 'logger' is defined and accessible in this scope if you uncomment the line below
        # import logging
        # logger = logging.getLogger(__name__) # Or use existing logger instance
        logger.info(
            f"Initialized SimplestDomainDiscriminator with input_total_channels: {input_total_channels}")

    def forward(self, features_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the simplest discriminator.
        Args:
            features_list: A list of feature tensors, one from each hooked layer.
        Returns:
            A tensor of logits [Batch, 1].
        """
        pooled_features = []
        for feat_map in features_list:
            if isinstance(feat_map, torch.Tensor):
                if feat_map.ndim == 2:
                    # This case assumes pre-flattened features for a single source, matching total channels
                    # If only one feature tensor is passed and it's already 2D [B, TotalChannels]
                    if feat_map.shape[1] == self.input_total_channels and len(features_list) == 1:
                        return self.model(feat_map) # Directly pass to model if it's a single, already correct feature
                    else:
                        # Fallback or error if multiple 2D features or mismatch without specific individual channel info
                        # For a truly "simplest" handling, one might assume it's already processed if 2D.
                        # However, to be consistent with multi-feature input, we'd expect concatenation later.
                        # Sticking to the robust multi-feature processing:
                        logger.warning(
                            f"SimplestD: Received 2D feature map of shape {feat_map.shape}. Appending as is. Ensure it's ready for concatenation if multiple features exist.")
                        pooled_features.append(feat_map)

                elif feat_map.ndim == 3:  # B, C, N (e.g., B, Channels, NumTokens)
                    # Average over the "spatial" or token dimension -> [B, C]
                    pooled = torch.mean(feat_map, dim=2)
                    pooled_features.append(pooled)
                elif feat_map.ndim == 4:  # B, C, H, W (typical CNN output)
                    pooled = self.pool(feat_map)  # Output shape: [B, C, 1, 1]
                    pooled_features.append(torch.flatten(pooled, 1))  # Output shape: [B, C]
                else:
                    logger.warning(
                        f"SimplestD: Skipping feature map with unexpected ndim: {feat_map.ndim}, shape: {feat_map.shape}")
                    continue
            else:
                logger.warning(f"SimplestD: Skipping non-Tensor input in features_list: {type(feat_map)}")
                continue

        if not pooled_features:
            logger.error("SimplestD: No valid features to process in discriminator after pooling/selection.")
            # Consider the batch size of the first input feature map if available, or use a placeholder if features_list was empty.
            # This path indicates an upstream issue.
            raise ValueError("SimplestD: Discriminator received no valid features to process.")

        try:
            # Concatenate pooled features along the channel dimension
            x = torch.cat(pooled_features, dim=1)  # Output shape: [B, total_channels]
        except Exception as e:
            logger.error(f"SimplestD: Error concatenating pooled features: {e}")
            for i, pf in enumerate(pooled_features):
                logger.error(f"SimplestD: Pooled feature {i} shape: {pf.shape}")
            raise

        # Validate the shape of the concatenated features
        if x.shape[1] != self.input_total_channels:
            logger.error(
                f"SimplestD: Discriminator input dimension mismatch! Expected {self.input_total_channels}, got {x.shape[1]}. "
                "This usually indicates an issue with hook setup or feature dimension calculation.")
            raise ValueError(
                f"SimplestD: Discriminator input dim mismatch: expected {self.input_total_channels}, got {x.shape[1]}")

        return self.model(x)




# --- Interface for users to add their own discriminators ---
def get_discriminator(name: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to get a discriminator.
    """
    if name == "default":
        hidden_dim = kwargs.get("hidden_dim", 1024)
        return DefaultDomainDiscriminator(input_total_channels=input_dim, hidden_dim=hidden_dim)
    elif name == "simpler":
        hidden_dim = kwargs.get("hidden_dim", 256)  # Default hidden_dim for simple
        return SimpleDomainDiscriminator(input_total_channels=input_dim, hidden_dim=hidden_dim)

    elif name == "simplest":
        return SimplestDomainDiscriminator(input_total_channels=input_dim)

    else:
        logger.error(f"Discriminator '{name}' not recognized.")
        raise ValueError(f"Discriminator '{name}' not recognized.")


if __name__ == '__main__':
    # (Your existing test code for discriminator_architectures.py)
    # ... (ensure it still works with LayerNorm)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():  # Add handler only if no handlers are configured
        logger.addHandler(handler)

    logger.info("Testing discriminator architectures with LayerNorm...")
    test_input_dim = 448
    discriminator = get_discriminator("default", input_dim=test_input_dim, hidden_dim=512)

    batch_size = 1  # Test with batch size 1 specifically
    dummy_features_bs1 = [
        torch.randn(batch_size, 64, 20, 20),
        torch.randn(batch_size, 128, 10, 10),
        torch.randn(batch_size, 256, 5, 5)
    ]

    try:
        output_logits = discriminator(dummy_features_bs1)
        logger.info(f"Discriminator output shape (BS=1): {output_logits.shape}")
        assert output_logits.shape == (batch_size, 1)
        logger.info("DefaultDomainDiscriminator with LayerNorm test (BS=1) successful.")
    except Exception as e:
        logger.error(f"Error during DefaultDomainDiscriminator with LayerNorm test (BS=1): {e}", exc_info=True)

    batch_size = 4  # Test with larger batch size
    dummy_features_bs4 = [
        torch.randn(batch_size, 64, 20, 20),
        torch.randn(batch_size, 128, 10, 10),
        torch.randn(batch_size, 256, 5, 5)
    ]
    try:
        output_logits_bs4 = discriminator(dummy_features_bs4)
        logger.info(f"Discriminator output shape (BS=4): {output_logits_bs4.shape}")
        assert output_logits_bs4.shape == (batch_size, 1)
        logger.info("DefaultDomainDiscriminator with LayerNorm test (BS=4) successful.")
    except Exception as e:
        logger.error(f"Error during DefaultDomainDiscriminator with LayerNorm test (BS=4): {e}", exc_info=True)
