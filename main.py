from src.workflow.training import training_pipeline
from src.utils.logger import get_logger
from src.utils.config import load_config


logger = get_logger(__name__)

def main():

    """
    Entry point for the application
    """

    training_pipeline()



    return None

if __name__ == "__main__":
    main()
