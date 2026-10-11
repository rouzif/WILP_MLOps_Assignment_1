import os
from pathlib import Path
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_up() -> None:
    """Deletes temporary data and artifact files configured in config.yaml."""
    logger.info("=== Starting Cleanup Pipeline ===")
    config = load_config()

    # List of configuration keys to clean up
    paths = [
        config['data']['input_data_path'],
        config['data']['column_category_path'],
        config['data']['missing_imputation_path'],
        config['data']['missing_imputed_data_path'],
        config['data']['column_category_encoded_path'],
        config['data']['encoded_data_path'],
        config['data']['column_category_scaled_path'],
        config['data']['scaling_params_path'],
        config['data']['scaled_data_path']
    ]

    for path_str in paths:

        file_path = Path(path_str)

        if file_path.exists():
            try:
                file_path.unlink()  # Deletes the file
                logger.info(f"Successfully deleted: {file_path}")
            except Exception as e:
                logger.error(
                    f"Failed to delete {file_path}: {e}", exc_info=True
                )
        else:
            logger.info(f"File does not exist, skipping: {file_path}")

    logger.info("=== Finished Cleanup Pipeline ===")

if __name__ == "__main__":
    clean_up()
