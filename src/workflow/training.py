import warnings
import mlflow, mlflow.sklearn
from src.data.load_data import run_data_load_pipeline
from src.data.partition import run_data_partitioning
from src.data.describe_data import run_data_description_pipeline
from src.data.visualize import run_visualization_pipeline
from src.data.preprocessor import run_preprocessor, DataPreprocessor
from src.model.train_model import run_training_pipeline
from src.model.predict import run_prediction_pipeline
from src.model.evaluation import run_evaluation_pipeline
from src.utils.data_fingerprint import get_data_fingerprint
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.utils.timestamp import get_timestamp


logger = get_logger(__name__)

def training_pipeline(run_id, algo):

    logger.info("\n\n")
    logger.info("Starting Machine Learning Training pipeline!\n\n")
    warnings.filterwarnings("ignore")

    # Loading config
    cfg = load_config()

    # Setting up Experiment tracking
    mlflow.set_tracking_uri(cfg['ml_flow']['tracking_uri'])
    mlflow.set_experiment(cfg['ml_flow']['experiment'])

    # Starting ML Flow Tracking
    with mlflow.start_run(run_name=run_id):

        # Loading data
        df = run_data_load_pipeline(cfg)

        # Logging data fingerprint
        mlflow.log_param("data_fingerprint", get_data_fingerprint(df))

        # Describe data
        column_types = run_data_description_pipeline(cfg, df)

        # Visualize data
        run_visualization_pipeline(cfg, df, column_types)

        # Partitioning data
        df_train, df_test = run_data_partitioning(cfg, df)

        # Logging model algorithm
        mlflow.log_param("model_kind", algo.upper())

        # Running preprocessing on training data
        df_train_processed, preprocessor = run_preprocessor(cfg, df_train)

        # Logging preprocessor
        mlflow.sklearn.log_model(
            sk_model=preprocessor,
            artifact_path="preprocessor",
            skops_trusted_types=['src.data.preprocessor.DataPreprocessor'],
        )

        # Running preprocessing on test data
        df_test_processed = preprocessor.transform(df_test)

        # Fitting model on train data
        model, model_params = run_training_pipeline(cfg, df_train_processed, preprocessor.column_types_updated_, algo)

        # Logging model params and model
        mlflow.log_params(model_params)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            skops_trusted_types=['sklearn.tree._tree.Tree'],
        )

        # Getting predictions on test data
        df_test_predictions = run_prediction_pipeline(cfg, df_test_processed, model, 'test')
        df_train_predictions = run_prediction_pipeline(cfg, df_train_processed, model, 'train')

        # Evaluation
        test_eval = run_evaluation_pipeline(cfg, df_test_predictions, 'test')
        train_eval = run_evaluation_pipeline(cfg, df_train_predictions, 'train')
        combined_metrics = {
            **{f"test_{k}": v for k, v in test_eval.items()},
            **{f"train_{k}": v for k, v in train_eval.items()},
        }

        # Logging metrics
        mlflow.log_metrics(combined_metrics)

        # Printing run details
        run = mlflow.active_run().info
        print(f"Logged run {run.run_id[:8]} auc={test_eval['roc_auc']:.4f}")

    logger.info("Machine Learning Training pipeline completed successfully!")

    return None

if __name__ == "__main__":
    training_pipeline(get_timestamp(), 'rf')
