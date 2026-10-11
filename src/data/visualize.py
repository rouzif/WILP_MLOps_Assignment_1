import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Apply a clean seaborn visual style
sns.set_theme(style="whitegrid")


def visualize_data(
    df: pd.DataFrame, column_types: dict, target_variable: str, output_dir: Path
) -> None:
    """Generates univariate, bivariate, and correlation heatmap visualizations and saves them to the configured directory."""
    df = df.copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Data Visualization ===")

    numeric_dist_cols = column_types.get("numeric-distributed", [])
    numeric_low_cols = column_types.get("numeric-low-cardinality", [])
    categorical_cols = column_types.get("categorical", [])

    low_cardinality_cols = numeric_low_cols + categorical_cols

    # -------------------------------------------------------------------------
    # 1. UNIVARIATE ANALYSIS
    # -------------------------------------------------------------------------
    logger.info("--- Generating Univariate Plots ---")

    # Histograms for each numeric-distributed column
    for col in numeric_dist_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True, color="skyblue")
            plt.title(f"Univariate Distribution - {col}", fontsize=12, fontweight="bold")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()

            plot_path = output_dir / f"univariate_hist_{col}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info(f"Saved histogram for '{col}' -> '{plot_path}'")

    # Bar charts for numeric-low-cardinality and categorical columns
    for col in low_cardinality_cols:
        if col in df.columns and col != target_variable:
            plt.figure(figsize=(8, 5))
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            sns.barplot(data=counts, x=col, y="count", palette="Blues_d")
            plt.title(f"Univariate Frequency - {col}", fontsize=12, fontweight="bold")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()

            plot_path = output_dir / f"univariate_bar_{col}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info(f"Saved bar chart for '{col}' -> '{plot_path}'")

    # Bar chart for target column
    if target_variable in df.columns:
        plt.figure(figsize=(6, 5))
        target_counts = df[target_variable].value_counts().reset_index()
        target_counts.columns = [target_variable, "count"]
        sns.barplot(data=target_counts, x=target_variable, y="count", palette="Set2")
        plt.title(f"Target Distribution - {target_variable}", fontsize=12, fontweight="bold")
        plt.xlabel(target_variable)
        plt.ylabel("Count")
        plt.tight_layout()

        target_plot_path = output_dir / f"target_distribution_{target_variable}.png"
        plt.savefig(target_plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved target distribution plot -> '{target_plot_path}'")
    else:
        logger.warning(f"Target variable '{target_variable}' not found in DataFrame.")

    # -------------------------------------------------------------------------
    # 2. BIVARIATE ANALYSIS
    # -------------------------------------------------------------------------
    logger.info("--- Generating Bivariate Plots ---")

    if target_variable in df.columns:
        # Numeric distributed: Box plots stratified by target variable
        for col in numeric_dist_cols:
            if col in df.columns:
                plt.figure(figsize=(8, 5))
                sns.boxplot(x=target_variable, y=col, data=df, palette="Set3")
                plt.title(f"Bivariate Analysis - {col} vs {target_variable}", fontsize=12, fontweight="bold")
                plt.xlabel(target_variable)
                plt.ylabel(col)
                plt.tight_layout()

                plot_path = output_dir / f"bivariate_box_{col}_vs_{target_variable}.png"
                plt.savefig(plot_path, dpi=300)
                plt.close()
                logger.info(f"Saved stratified box plot for '{col}' -> '{plot_path}'")

        # Numeric low cardinality and categorical: Bar chart depicting %(Target = 1) for each entry
        for col in low_cardinality_cols:
            if col in df.columns and col != target_variable:
                plt.figure(figsize=(8, 5))
                target_pct = (
                    df.groupby(col)[target_variable]
                    .mean()
                    .reset_index()
                )
                target_pct[target_variable] = target_pct[target_variable] * 100

                sns.barplot(
                    data=target_pct,
                    x=col,
                    y=target_variable,
                    palette="viridis",
                )
                plt.title(
                    f"Bivariate Analysis - %({target_variable} = 1) by {col}",
                    fontsize=12,
                    fontweight="bold",
                )
                plt.xlabel(col)
                plt.ylabel(f"% ({target_variable} = 1)")
                plt.tight_layout()

                plot_path = output_dir / f"bivariate_pct_{col}_vs_{target_variable}.png"
                plt.savefig(plot_path, dpi=300)
                plt.close()
                logger.info(
                    f"Saved target percentage bar chart for '{col}' -> '{plot_path}'"
                )

    # -------------------------------------------------------------------------
    # 3. CORRELATION HEATMAP
    # -------------------------------------------------------------------------
    logger.info("--- Generating Correlation Heatmap ---")

    # Combine all numerical columns (distributed + low cardinality + target)
    all_numeric_cols = list(
        set(numeric_dist_cols + numeric_low_cols + ([target_variable] if target_variable in df.columns else []))
    )
    valid_numeric_cols = [col for col in all_numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if len(valid_numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[valid_numeric_cols].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Correlation Heatmap (Numeric Variables & Target)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        heatmap_path = output_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        logger.info(f"Saved correlation heatmap -> '{heatmap_path}'")
    else:
        logger.warning("Insufficient numeric variables present to create a correlation heatmap.")

    logger.info("=== Finished Data Visualization ===")


def run_visualization_pipeline(cfg: dict, df: pd.DataFrame, column_types: dict) -> None:
    """Executes the visualization step using configured paths."""
    logger.info("\n\n")
    logger.info("=== Running Data Visualization Pipeline ===")

    target_var = cfg['variables']['target']
    vis_dir = Path(cfg["data"]["visualisation_directory"])

    visualize_data(
        df=df,
        column_types=column_types,
        target_variable=target_var,
        output_dir=vis_dir,
    )

    logger.info("=== Finished Visualization Pipeline Execution ===")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logger.info("Initializing Visualization Pipeline Script Execution...")

    # Load config
    cfg = load_config()

    # Load dataframe
    input_data_path = cfg["data"]["input_data_path"]
    logger.info(f"Loading data for visualization from: '{input_data_path}'")
    df = pd.read_csv(input_data_path)

    # Load column types dictionary
    cat_path = cfg["data"]["column_category_path"]
    logger.info(f"Loading column category metadata from: '{cat_path}'")
    with open(cat_path, "r", encoding="utf-8") as f:
        col_types = json.load(f)

    # Execute visualization pipeline
    run_visualization_pipeline(cfg, df, col_types)

    logger.info("Script execution completed successfully.")