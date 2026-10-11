from src.data.load_data import run_data_load_pipeline
from src.data.describe_data import describe_data
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
    column_types = describe_data(
        df=df,
    )

    logger.info("Machine Learning pipeline completed successfully!")

    return None

if __name__ == "__main__":
    main()
