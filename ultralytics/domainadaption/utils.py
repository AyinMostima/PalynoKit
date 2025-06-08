import logging
import sys
from pathlib import Path
import shutil
import random
import torch
import yaml

# --- Constants ---
SOURCE_DOMAIN_LABEL = 1.0
TARGET_DOMAIN_LABEL = 0.0


# --- Logging Setup ---
def setup_logging(log_file_path: Path = None, level=logging.INFO):
    """
    Sets up basic logging configuration.
    """
    log_format = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    logging.info("Logging setup complete.")


# --- File/Directory Operations ---
def ensure_dir_exists(dir_path: Path):
    """Ensures a directory exists, creates it if not."""
    dir_path.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Ensured directory exists: {dir_path}")


def copy_files_from_list(file_list: list, destination_dir: Path, num_files_to_copy: int = -1):
    """
    Copies a specified number of files from a list to a destination directory.
    If num_files_to_copy is -1, copies all files.
    """
    ensure_dir_exists(destination_dir)

    if not file_list:
        logging.warning(f"No files found in the source list to copy to {destination_dir}.")
        return

    if num_files_to_copy != -1 and len(file_list) > num_files_to_copy:
        files_to_process = random.sample(file_list, num_files_to_copy)
        logging.info(f"Randomly selected {num_files_to_copy} files for copying.")
    else:
        files_to_process = file_list
        if num_files_to_copy != -1:  # Means len(file_list) <= num_files_to_copy
            logging.warning(
                f"Requested {num_files_to_copy} files, but only {len(file_list)} available. Copying all available.")

    copied_count = 0
    for src_file in files_to_process:
        try:
            shutil.copy(src_file, destination_dir / src_file.name)
            copied_count += 1
        except Exception as e:
            logging.error(f"Error copying file {src_file} to {destination_dir}: {e}")
    logging.info(f"Copied {copied_count}/{len(files_to_process)} files to {destination_dir}")


def get_image_files(image_dir: Path) -> list:
    """Returns a list of image files (jpg, jpeg, png) from a directory."""
    if not image_dir.is_dir():
        logging.error(f"Image directory not found: {image_dir}")
        return []
    return list(image_dir.glob('*.[jp][pn]g')) + list(image_dir.glob('*.jpeg'))


# --- Model Related Utilities ---
def get_device(device_str: str = "auto") -> torch.device:
    """Gets the torch device."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logging.info(f"Using device: {device}")
    return device


# --- YAML parsing for model structure (as requested) ---
def parse_ultralytics_yaml(yaml_path: Path) -> dict:
    """
    Parses an Ultralytics YAML model configuration file.
    Returns the parsed dictionary.
    """
    if not yaml_path.exists():
        logging.error(f"YAML configuration file not found: {yaml_path}")
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        try:
            cfg = yaml.safe_load(f)
            logging.info(f"Successfully parsed YAML file: {yaml_path}")
            return cfg
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {yaml_path}: {e}")
            raise


def extract_backbone_definition_from_cfg(model_cfg: dict) -> list | None:
    """
    Extracts the 'backbone' definition from a parsed Ultralytics model config.
    Assumes 'backbone' is a top-level key with a list of layer definitions.
    """
    if 'backbone' in model_cfg and isinstance(model_cfg['backbone'], list):
        logging.info(f"Extracted 'backbone' definition with {len(model_cfg['backbone'])} layers.")
        return model_cfg['backbone']
    else:
        logging.warning("'backbone' key not found or not a list in model configuration.")
        return None


if __name__ == '__main__':
    # Example usage (optional, for testing this module)
    setup_logging()
    logging.info("utils.py executed as main.")
    test_yaml_path = Path(r"path_to_your_model.yaml")  # Replace with a valid path for testing
    if test_yaml_path.exists():
        cfg = parse_ultralytics_yaml(test_yaml_path)
        if cfg:
            backbone_def = extract_backbone_definition_from_cfg(cfg)
            if backbone_def:
                logging.info(f"First backbone layer: {backbone_def[0]}")
    else:
        logging.warning(f"Test YAML {test_yaml_path} not found, skipping YAML parsing test.")
