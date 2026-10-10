import pandas as pd
from ucimlrepo import fetch_ucirepo
from src.utils.logger import get_logger
from src.utils.config import load_config


logger = get_logger(__name__)

def load_data(config: dict) -> None:


    logger.info("Loading data...")

    # Loading data
    if config['data']['load_from_saved_file']:

        df = pd.read_csv(config['data']['load_from_saved_file'])

    else:

        heart_disease = fetch_ucirepo(id=45)
        df = heart_disease.data.features
        df['heart_disease'] = heart_disease.data.targets

    # Logging important stats
    logger.info(f"Number of rows: {df.shape[0]}.")
    logger.info(f"Number of columns: {df.shape[1]}.")
    logger.info(f"Variables: {', '.join(df.columns)}.")

    logger.info("Data loading complete.")

    # Saving data
    save_file_path = config['data']['saved_file_path']
    if not config['data']['load_from_saved_file'] and config['data']['save_loaded_data']:
        logger.info(f"Saving data...")
        df.to_csv(save_file_path, index=False)
        logger.info(f"Data saved to {save_file_path}.")

    return df


if __name__ == "__main__":

    cfg = load_config()
    df = load_data(cfg)
