import argparse
import sys
from src.workflow.training import training_pipeline
from src.workflow.inference import inference_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Entry point for the application."""
    parser = argparse.ArgumentParser(description="ML Pipeline Execution Entry Point")
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=["train", "infer"],
        help="Specify which pipeline to execute: 'train' or 'infer'",
    )

    args = parser.parse_args()

    if args.pipeline == "train":
        logger.info("Running Training Pipeline...")
        training_pipeline()
    elif args.pipeline == "infer":
        logger.info("Running Inference Pipeline...")
        inference_pipeline()
    else:
        logger.error(f"Invalid pipeline option: {args.pipeline}. Running Training Pipeline...")
        training_pipeline()

    return None


if __name__ == "__main__":
    main()