import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import concurrent.futures
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# ----------------------
# Helpers
# ----------------------
def _safe_inverse_transform(enc: Optional[OrdinalEncoder], df_cat: pd.DataFrame) -> pd.DataFrame:
    """
    Safely inverse-transforms encoded categorical features to their original values.

    Args:
        enc: OrdinalEncoder instance used for encoding, or None if no encoding was applied.
        df_cat: DataFrame with encoded categorical features.

    Returns:
        DataFrame with original categorical values, or encoded values if decoding fails.
    """
    if enc is None or df_cat.empty:
        return df_cat
    try:
        return pd.DataFrame(enc.inverse_transform(df_cat), columns=df_cat.columns, index=df_cat.index)
    except Exception:
        # Fallback to encoded values if inverse transform fails (e.g., unknown values)
        return df_cat


def _impute_and_score(
    imputer: Any,
    X_num: pd.DataFrame,
    X_cat_enc: pd.DataFrame,
    y: pd.Series,
    model: Any,
    metric: Union[str, Any],
    cv: int,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> Tuple[float, float, pd.DataFrame, pd.DataFrame]:
    """
    Imputes missing values and evaluates the strategy using cross-validation.

    Args:
        imputer: Scikit-learn imputer for numeric features (e.g., SimpleImputer, KNNImputer).
        X_num: DataFrame with numeric features.
        X_cat_enc: DataFrame with encoded categorical features.
        y: Target series.
        model: Scikit-learn model for evaluation.
        metric: Scoring metric for cross-validation (e.g., 'accuracy', 'neg_mean_squared_error').
        cv: Number of cross-validation folds.
        categorical_cols: List of categorical column names.
        numeric_cols: List of numeric column names.

    Returns:
        Tuple of (mean cross-validation score, standard deviation of scores,
                 imputed numeric DataFrame, imputed categorical DataFrame).
    """
    # Handle numeric imputation
    X_num_imp = pd.DataFrame(
        imputer.fit_transform(X_num) if not X_num.empty else [],
        columns=numeric_cols,
        index=X_num.index
    )

    # Handle categorical imputation with most-frequent strategy
    X_cat_imp = pd.DataFrame(
        SimpleImputer(strategy="most_frequent").fit_transform(X_cat_enc) if not X_cat_enc.empty else [],
        columns=categorical_cols,
        index=X_cat_enc.index
    )

    # Combine imputed features
    X_imp = pd.concat([X_num_imp, X_cat_imp], axis=1)

    # Clone model to avoid state contamination
    model_local = clone(model)
    scores = cross_val_score(model_local, X_imp, y, cv=cv, scoring=metric)
    return float(np.mean(scores)), float(np.std(scores)), X_num_imp, X_cat_imp


# ----------------------
# Main function
# ----------------------
def auto_impute(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    models: Optional[List[Any]] = None,
    metric: Optional[Union[str, Any]] = None,
    cv: int = 5,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    verbose: bool = True,
    custom_strategies: Optional[Dict[str, Any]] = None,
    visualize: bool = True,
    save_plot: bool = False,
    plot_path: Optional[str] = None,
    timeout_sec: int = 60,
    feature_limit: Optional[int] = None,
    return_best_only: bool = False,
    n_jobs: int = 1,
    use_process_pool: bool = True,
    save_best_path: Optional[str] = None,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Automatically evaluates imputation strategies for missing values using cross-validation.
    Selects the best strategy based on model performance.

    Args:
        df: Input DataFrame with potential missing values.
        target_col: Name of the target column (must not contain missing values).
        numeric_cols: List of numeric column names. If None, auto-detected.
        categorical_cols: List of categorical column names. If None, auto-detected.
        models: List of scikit-learn models for evaluation. Defaults to LogisticRegression for
                classification or LinearRegression for regression.
        metric: Scoring metric for cross-validation (e.g., 'accuracy', 'neg_mean_squared_error').
                Defaults to 'accuracy' for classification, 'neg_mean_squared_error' for regression.
        cv: Number of cross-validation folds. Defaults to 5.
        sample_frac: Fraction of data to sample for faster processing. Defaults to None (use all data).
        random_state: Seed for reproducibility. Defaults to 42.
        verbose: If True, logs progress and results. Defaults to True.
        custom_strategies: Dictionary of additional imputation strategies (e.g., {'name': imputer}).
        visualize: If True, plots a comparison of imputation strategies. Defaults to True.
        save_plot: If True, saves the plot to `plot_path`. Defaults to False.
        plot_path: File path for saving the plot. Defaults to None.
        timeout_sec: Timeout per strategy evaluation in seconds. Defaults to 60.
        feature_limit: Maximum number of features to process. Defaults to None (use all).
        return_best_only: If True, returns only the best DataFrame and info. Defaults to False.
        n_jobs: Number of parallel jobs. Defaults to 1.
        use_process_pool: If True, attempts to use ProcessPoolExecutor; falls back to ThreadPoolExecutor
                         if serialization fails. Defaults to True.
        save_best_path: File path to save the best imputed DataFrame as CSV. Defaults to None.

    Returns:
        If return_best_only=True: (best imputed DataFrame, best strategy info dict).
        Otherwise: (best imputed DataFrame, results DataFrame, best strategy info dict).
        The info dict contains: strategy name, model name, score, standard deviation, and execution time.

    Raises:
        ValueError: If target column is missing or contains NaN values.
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Configure logging
    logger = logging.getLogger("auto_impute")
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="[%(levelname)s] %(message)s")

    # Validate inputs
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if df[target_col].isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")

    # Auto-detect numeric and categorical columns
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors="ignore").tolist()
    numeric_cols = [c for c in numeric_cols if not np.issubdtype(df[c].dtype, np.datetime64)]

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.drop(target_col, errors="ignore").tolist()

    # Determine task type (classification or regression)
    target_is_classification = df[target_col].dtype == "object" or df[target_col].nunique() < 15

    # Set default models and metric
    if models is None:
        models = [LogisticRegression(max_iter=1000, solver="liblinear")] if target_is_classification else [LinearRegression()]
    if metric is None:
        metric = "accuracy" if target_is_classification else "neg_mean_squared_error"

    # Define imputation strategies
    strategies: Dict[str, Any] = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "most_frequent": SimpleImputer(strategy="most_frequent"),
        "constant_0": SimpleImputer(strategy="constant", fill_value=0),
        "iterative": IterativeImputer(max_iter=10, random_state=random_state, sample_posterior=False),
    }
    for k in [3, 5, 7]:
        strategies[f"knn_{k}"] = KNNImputer(n_neighbors=k)
    if custom_strategies:
        strategies.update(custom_strategies)

    # Apply sampling and feature limits
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=random_state)
    if feature_limit and len(numeric_cols) > feature_limit:
        numeric_cols = numeric_cols[:feature_limit]
        logger.info(f"⚠ Limited to {feature_limit} features due to feature_limit.")

    # Split data into numeric, categorical, and target
    X_num = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
    X_cat = df[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=df.index)
    y = df[target_col].copy()

    # Encode categorical features
    if not X_cat.empty:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat_enc = pd.DataFrame(enc.fit_transform(X_cat), columns=categorical_cols, index=X_cat.index)
    else:
        enc = None
        X_cat_enc = pd.DataFrame(index=X_num.index)

    # Initialize results storage
    results: Dict[str, Dict[str, Any]] = {}
    best_info: Dict[str, Any] = {"score": -np.inf, "strategy": None, "model": None, "std": None, "time_sec": None}
    best_df = df.copy()

    # Setup progress bar
    total_tasks = len(strategies) * len(models)
    pbar = tqdm(total=total_tasks, desc="Testing strategies", ncols=100) if verbose else None

    # Choose executor (ProcessPool with fallback to ThreadPool)
    executor_cls = ThreadPoolExecutor
    used_process_pool = False
    if use_process_pool:
        try:
            # Test ProcessPoolExecutor feasibility
            tmp = ProcessPoolExecutor(max_workers=min(n_jobs, max(1, total_tasks)))
            tmp.shutdown(wait=True)
            executor_cls = ProcessPoolExecutor
            used_process_pool = True
        except Exception as e:
            logger.warning("ProcessPoolExecutor unavailable (serialization issue). Using ThreadPoolExecutor. (%s)", e)
            executor_cls = ThreadPoolExecutor
            used_process_pool = False

    # Submit imputation tasks
    futures_map: Dict[concurrent.futures.Future, Tuple[str, str, float]] = {}
    try:
        with executor_cls(max_workers=n_jobs) as executor:
            for name, imputer in strategies.items():
                for model in models:
                    model_clone = clone(model)
                    try:
                        fut = executor.submit(
                            _impute_and_score,
                            imputer,
                            X_num,
                            X_cat_enc,
                            y,
                            model_clone,
                            metric,
                            cv,
                            categorical_cols,
                            numeric_cols,
                        )
                        futures_map[fut] = (name, model.__class__.__name__, time.time())
                    except Exception as e:
                        key = f"{name} | {model.__class__.__name__}"
                        results[key] = {"error": f"Submit failed: {e}"}
                        logger.error("Submit failed for %s: %s", key, e)
                        if pbar:
                            pbar.update(1)

            # Collect results
            for fut in concurrent.futures.as_completed(futures_map, timeout=max(1, timeout_sec) * len(futures_map)):
                name, model_name, start_time = futures_map[fut]
                key = f"{name} | {model_name}"
                if pbar:
                    pbar.update(1)
                try:
                    mean_score, std_score, X_num_imp, X_cat_imp = fut.result(timeout=timeout_sec)
                    elapsed = round(time.time() - start_time, 2)
                    results[key] = {"score": mean_score, "std": std_score, "time_sec": elapsed}

                    # Update best result
                    if mean_score > best_info["score"]:
                        best_info.update({"strategy": name, "model": model_name, "score": mean_score, "std": std_score, "time_sec": elapsed})
                        if not X_num_imp.empty:
                            best_df.loc[:, numeric_cols] = X_num_imp
                        if not X_cat_enc.empty:
                            best_df.loc[:, categorical_cols] = _safe_inverse_transform(enc, X_cat_imp)

                except concurrent.futures.TimeoutError:
                    results[key] = {"error": f"Timeout ({timeout_sec}s)"}
                    logger.warning("Timeout for %s", key)
                except Exception as e:
                    results[key] = {"error": str(e)}
                    logger.error("Error for %s: %s", key, e)

    except concurrent.futures.TimeoutError:
        logger.error("Global timeout while waiting for futures.")
    except Exception as e:
        logger.error("Executor error: %s", e)

    if pbar:
        pbar.close()

    if best_info.get("strategy") is None:
        logger.warning("No strategy returned a valid result.")

    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    if "score" in results_df.columns:
        results_df = results_df.sort_values("score", ascending=False)
        baseline = results_df["score"].iloc[-1]
        results_df["improvement_%"] = 100 * (results_df["score"] - baseline) / (abs(baseline) + 1e-12)

    # Visualize results
    if visualize and "score" in results_df.columns and not results_df.empty:
        plt.figure(figsize=(8, max(4, 0.2 * len(results_df))))
        display_df = results_df.sort_values("score")
        plt.barh(display_df.index, display_df["score"], xerr=display_df["std"])
        plt.title("Imputation Strategy Comparison")
        plt.xlabel("Cross-validated Score")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_plot and plot_path:
            plt.savefig(plot_path, dpi=300)
        plt.show()

    # Save best DataFrame if requested
    if save_best_path and best_df is not None:
        try:
            best_df.to_csv(save_best_path, index=False)
            logger.info("Best DataFrame saved to %s", save_best_path)
        except Exception as e:
            logger.warning("Failed to save best_df: %s", e)

    if verbose and best_info.get("strategy") is not None:
        logger.info("Best strategy: %s (%s) | Score: %.6f", best_info["strategy"], best_info["model"], float(best_info["score"]))

    return (best_df, best_info) if return_best_only else (best_df, results_df, best_info)


# Example usage
# if __name__ == "__main__":
#     import sklearn.datasets
#     X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
#     df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
#     df["target"] = y
#     df.loc[df.sample(frac=0.1, random_state=1).index, "f0"] = np.nan
#     best_df, results_df, best_info = auto_impute(
#         df, "target", n_jobs=2, verbose=True, save_best_path="best_imputed.csv"
#     )
#     print(best_info)