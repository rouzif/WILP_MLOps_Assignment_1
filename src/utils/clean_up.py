import os
from pathlib import Path
import shutil
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_up() -> None:
    """Deletes temporary data, artifact files, and directories configured in config.yaml."""
    logger.info("=== Starting Cleanup Pipeline ===")
    config = load_config()

    # List of configuration paths to clean up
    paths = [
        config["data"]["input_data_path"],
        config["data"]["column_category_path"],
        config["data"]["visualisation_directory"],
        config["data"]["missing_imputation_path"],
        config["data"]["missing_imputed_data_path"],
        config["data"]["column_category_encoded_path"],
        config["data"]["encoded_data_path"],
        config["data"]["column_category_scaled_path"],
        config["data"]["scaling_params_path"],
        config["data"]["scaled_data_path"],
        config["data"]["encoding_scheme_path"],
        config["data"]["train_data_path"],
        config["data"]["test_data_path"],
        config["data"]["preprocessor_path"],
        config["model"]["model_path"],
        config["data"]["predicted_path_prefix"] + 'test.csv',
        config["data"]["predicted_path_prefix"] + 'train.csv',
        config["data"]["predicted_path_prefix"] + '.csv',
        config["model"]["eval_path_prefix"] + 'test.json',
        config["model"]["eval_path_prefix"] + 'train.json',
        config["model"]["eval_path_prefix"] + '.json',
    ]

    for path_str in paths:
        file_path = Path(path_str)

        if file_path.exists():
            try:
                if file_path.is_dir():
                    # Delete directory and all its contents
                    shutil.rmtree(file_path)
                    logger.info(
                        f"Successfully deleted directory and contents: {file_path}"
                    )
                elif file_path.is_file():
                    # Delete single file
                    file_path.unlink()
                    logger.info(f"Successfully deleted file: {file_path}")
            except Exception as e:
                logger.error(
                    f"Failed to delete {file_path}: {e}", exc_info=True
                )
        else:
            logger.info(f"Path does not exist, skipping: {file_path}")

    logger.info("=== Finished Cleanup Pipeline ===")


if __name__ == "__main__":
    clean_up()