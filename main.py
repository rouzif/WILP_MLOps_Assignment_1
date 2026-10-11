from src.data.load_data import run_data_load_pipeline
from src.data.describe_data import run_data_description_pipeline
from src.data.missing_values import run_missing_value_pipeline
from src.data.encode import run_encoding_pipeline
from src.utils.logger import get_logger
from src.utils.config import load_config


logger = get_logger(__name__)

def main():

    """
    Entry point for the application
    """

    logger.info("Starting Machine Learning pipeline!\n\n")

    # Loading config
    cfg = load_config()

    # Loading data
    df = run_data_load_pipeline(cfg)

    # Describe data
    column_types = run_data_description_pipeline(cfg, df)

    # Missing value imputation
    df, missing_imputation = run_missing_value_pipeline(cfg, df, column_types)

    # Encoding
    df, column_types = run_encoding_pipeline(cfg, df, column_types)

    logger.info("Machine Learning pipeline completed successfully!")

    return None

if __name__ == "__main__":
    main()
