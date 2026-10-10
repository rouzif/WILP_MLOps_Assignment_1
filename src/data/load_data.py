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

    logger.info(f"Variables: {', '.join(df.columns)}")

    logger.info("Data loading complete.")

    # Saving data
    if not config['data']['load_from_saved_file'] and config['data']['save_loaded_data']:
        df.to_csv("data/processed/cleaned.csv", index=False)

    return None

if __name__ == "__main__":

    cfg = load_config()
    load_data(cfg)
