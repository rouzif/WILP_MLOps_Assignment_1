import argparse
import sys
from src.workflow.training import training_pipeline
from src.workflow.inference import inference_pipeline
from src.utils.logger import get_logger
from src.utils.timestamp import get_timestamp

logger = get_logger(__name__)


def main():
    """Entry point for the application."""

    # Parsing arguments
    parser = argparse.ArgumentParser(description="ML Pipeline Execution Entry Point")
    parser.add_argument(
        "--pipeline",
        type=str,
        required=False,
        choices=["train", "infer"],
        default="train",
        help="Specify which pipeline to execute: 'train' or 'infer'",
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=False,
        choices=["rf", "xgb"],
        default="rf",
        help="Specify which model training algorithm to use: 'rf' or 'xgb'",
    )

    args = parser.parse_args()

    run_id = get_timestamp()

    if args.pipeline == "train":
        logger.info("Running Training Pipeline...")
        training_pipeline(run_id, args.algo)
    elif args.pipeline == "infer":
        logger.info("Running Inference Pipeline...")
        inference_pipeline()
    else:
        logger.error(f"Invalid pipeline option: {args.pipeline}.")
        sys.exit(1)

    return None


if __name__ == "__main__":
    main()