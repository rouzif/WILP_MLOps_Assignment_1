import warnings
from src.data.load_data import run_data_load_pipeline
from src.data.partition import run_data_partitioning
from src.data.describe_data import run_data_description_pipeline
from src.data.visualize import run_visualization_pipeline
from src.data.preprocessor import run_preprocessor
from src.model.train_model import run_training_pipeline
from src.model.predict import run_prediction_pipeline
from src.utils.logger import get_logger
from src.utils.config import load_config


logger = get_logger(__name__)

def main():

    """
    Entry point for the application
    """

    logger.info("Starting Machine Learning pipeline!\n\n")
    warnings.filterwarnings("ignore")

    # Loading config
    cfg = load_config()

    # Loading data
    df = run_data_load_pipeline(cfg)

    # Describe data
    column_types = run_data_description_pipeline(cfg, df)

    # Visualize data
    run_visualization_pipeline(cfg, df, column_types)

    # Partitioning data
    df_train, df_test = run_data_partitioning(cfg, df)

    # Running preprocessing on training data
    df_train_processed, preprocessor = run_preprocessor(cfg, df_train)

    # Running preprocessing on test data
    df_test_processed = preprocessor.transform(df_test)

    # Fitting model on train data
    model = run_training_pipeline(cfg, df_train_processed, preprocessor.column_types_updated_)

    # Getting predictions on test data
    df_test_predictions = run_prediction_pipeline(cfg, df_test_processed, model, 'test')
    df_train_predictions = run_prediction_pipeline(cfg, df_test_processed, model, 'train')

    # Evaluating on test data

    logger.info("Machine Learning pipeline completed successfully!")

    return None

if __name__ == "__main__":
    main()
