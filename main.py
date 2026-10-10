from src.data.load_data import load_data
from src.utils.logger import get_logger
from src.utils.config import load_config


logger = get_logger(__name__)

def main():

    """
    Entry point for the application
    """

    logger.info("Starting Machine Learning pipeline!")

    # Loading config
    cfg = load_config()

    # Loading data
    df = load_data(cfg)

    logger.info("Machine Learning pipeline completed successfully!")

    return None

if __name__ == "__main__":
    main()
