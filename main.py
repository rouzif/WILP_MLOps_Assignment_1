from src.data.load_data import load_data
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
    df = load_data(
        load_from_saved_file=cfg['data']['load_from_saved_file'],
        saved_file_path=cfg['data']['input_data_path']
    )

    # Describe data
    df = describe_data(
        df=df,
    )

    logger.info("Machine Learning pipeline completed successfully!")

    return None

if __name__ == "__main__":
    main()
