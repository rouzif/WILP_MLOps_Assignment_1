import numpy as np
import pandas as pd
from typing import Optional
from ucimlrepo import fetch_ucirepo
from src.utils.logger import get_logger
from src.utils.config import load_config


logger = get_logger(__name__)

def load_data(load_from_saved_file: bool, saved_file_path: str = '') -> pd.DataFrame:


    logger.info("Loading data...")

    # Loading data
    if load_from_saved_file:

        df = pd.read_csv(saved_file_path)

    else:

        heart_disease = fetch_ucirepo(id=45)
        df = heart_disease.data.features
        df['heart_disease'] = heart_disease.data.targets

    # Logging important stats
    logger.info(f"Number of rows: {df.shape[0]}.")
    logger.info(f"Number of columns: {df.shape[1]}.")
    logger.info(f"Variables: {', '.join(df.columns)}.")

    logger.info("Data loading complete.\n")

    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:

    logger.info(f"Saving data...")
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to {output_path}.\n")


def refactor_data(df: pd.DataFrame) -> None:

    logger.info(f"Refactoring data...")
    df['heart_disease_flag'] = np.where(df['heart_disease'] > 0, 1, 0)
    df = df.drop(columns=['heart_disease'])

    return df


def run_data_load_pipeline(cfg: dict) -> pd.DataFrame:

    df = load_data(
        load_from_saved_file=cfg['data']['load_from_saved_file'],
        saved_file_path=cfg['data']['input_data_path']
    )

    df = refactor_data(
        df=df
    )

    if cfg['data']['save_loaded_data']:
        save_data(
            df=df,
            output_path=cfg['data']['input_data_path'],
        )

    return df

if __name__ == "__main__":

    cfg = load_config()
    df = run_data_load_pipeline(cfg)
