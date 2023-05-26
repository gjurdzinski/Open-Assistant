from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
from utils import add_labeler_id_column
import pandas as pd
import numpy as np
import ast


def get_buckets_bounds_and_scaler(
    dataset_df: pd.DataFrame,
    tr_preds_df: pd.DataFrame,
    metric: str,
) -> Tuple[List[float], MinMaxScaler]:
    """
    Returns the bounds for the buckets for the given metric.
    Args:
        dataset_df: DataFrame as loaded from newsroom csv. Should contain
            columns for requested metric (metric itself with all ratings and
            split), `ArticleID` and `System`.
        tr_preds_df: DataFrame with predictions for the training split. Should
            contain columns `ArticleID`, `System` and `deberta_reward`.
        metric: Name of the metric for which to optimize the buckets.
    Returns:
        bounds: List of bounds for the buckets.
        scaler: Scaler to use for scaling the model's reward output.
    """
    tr_df_wide = _get_tr_df_wide(dataset_df, tr_preds_df, metric=metric)

    # fit normalizer
    scaler = MinMaxScaler(clip=True)
    scaler.fit(tr_df_wide["deberta_reward"].to_numpy().reshape(-1, 1))

    # transform rewards on training split
    tr_df_wide["deberta_reward_scaled"] = scaler.transform(
        tr_df_wide["deberta_reward"].to_numpy().reshape(-1, 1)
    )

    # fit bucketer
    bounds, _ = optimize_buckets(
        tr_df_wide,
        optimization_column="deberta_reward_scaled",
        human_prefix="labeler_",
    )
    return bounds, scaler


def map_to_optimal_buckets(
    df: pd.DataFrame,
    bounds: List[float],
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    """
    Maps the model's reward output into labels.
    Args:
        df: DataFrame where each row is a one example and each column is
            an annotator. The values of human labels should not be normalized.
        bounds: List of bounds for the buckets.
        scaler: Scaler to use for scaling the model's reward output.
        metric: Name of the metric for which to optimize the buckets.
    Returns:
        df: DataFrame with added column `labels` which contains the labels
            after mapping to optimal buckets.
    """
    df = df.copy()
    # transform rewards on test split
    df["deberta_reward_scaled"] = scaler.transform(
        df["deberta_reward"].to_numpy().reshape(-1, 1)
    )

    # map to buckets
    df["labels"] = df["deberta_reward_scaled"].apply(
        lambda x: map_to_optimal_bucket(x, bounds)
    )
    return df


def map_to_optimal_bucket(x: float, boundaries: List[float]) -> int:
    """
    Maps model value to int label by provided boundaries.
    Args:
        x: value which has to be mapped
        boundaries: any long list of floats which describes boundaries of
            mapping buckets e.g. [0.33, 0.54] where
            label 1 <= 0.33 < label 2 <= 0.54 < label 3
    """
    for idx, bound in enumerate(boundaries):
        if x <= bound:
            return idx + 1
    return len(boundaries) + 1


def optimize_buckets(
    df: pd.DataFrame,
    optimization_column: str,
    human_prefix: str = "label_",
    n_labels: int = 5,
    verbose: bool = False,
) -> Tuple[List[float], float]:
    """
    Oprimizes mapping buckets for n labels for given models continuous outputs.
    Args:
        df: DataFrame where each row is a one example and each column is
            an annotator. The values of human labels should not be normalized.
        optimization_column: column with values under which we want to
            optimize labels mapping (values should be normalized to range 0-1)
        human_prefix: prefix of the columns' names that store the reference
            annotations (human labels).
        n_labels: number of labels to which we want to map the model's
            responses.
        verbose: whether to print optimization progress.
    Returns:
        boundaries: List of bounds for the buckets.
        max_score: The score of the best mapping.
    """
    df = df.copy()
    max_score = -np.inf
    previous_bound = 0
    boundaries = []
    for bound_num in range(n_labels - 1):
        if verbose:
            print(f"Optimizing {bound_num + 1} boundary")
        temp_boundaries = boundaries.copy()
        temp_boundaries.append(previous_bound)
        for bound in np.arange(previous_bound, 1, 0.01):
            temp_boundaries[bound_num] = bound
            df["model"] = df[optimization_column].apply(
                lambda x: map_to_optimal_bucket(x, temp_boundaries)
            )
            score = _get_labeler_score_optimization_version(
                df, "model", human_prefix, len(temp_boundaries) + 1
            )
            if bound_num == 0:
                if score >= max_score:
                    if len(boundaries) == bound_num:
                        boundaries.append(bound)
                    else:
                        boundaries[bound_num] = bound
                    max_score = score
            else:
                if score > max_score:
                    if len(boundaries) == bound_num:
                        boundaries.append(bound)
                    else:
                        boundaries[bound_num] = bound
                    max_score = score
        max_score = -np.inf
        previous_bound = boundaries[bound_num] + 0.01
    return boundaries, max_score


def _get_labeler_score_optimization_version(
    df: pd.DataFrame,
    labeler_col: str,
    human_prefix: str = "label_",
    n_labels: int = 5,
):
    """
    Simulates evaluation scoring but for given labels number (n_labels). Used
    for mapping buckets optimization.
    Args:
        df: DataFrame where each row is a one example and each column is
            an annotator. The values should not be normalized.
        labeler_col: name of column which contains continous values of model
            mapped to human labels
        human_prefix: prefix of the columns names that store the reference
            annotations (human labels)
        n_labels: number of labels which were considered during mapping
    """
    comparison_cols = [
        col
        for col in df.columns
        if (col.startswith(human_prefix) and col != labeler_col)
    ]
    scores = []
    for _, row in df.iterrows():
        labeler_score = row[labeler_col]
        comparison_scores = row[comparison_cols].values.tolist()
        if labeler_score > n_labels:
            labeler_score = n_labels
        comparison_scores = [
            n_labels if score > n_labels else score
            for score in comparison_scores
        ]
        if not np.isnan(comparison_scores).all():
            typical_score = np.nanmedian(comparison_scores)
            score = (
                1
                - ((typical_score - labeler_score) ** 2) / (n_labels - 1) ** 2
            )
            scores.append(score)
        else:
            scores.append(np.nan)
    return np.nanmean(scores)


def _get_tr_df_wide(dataset_df, preds_df, metric):
    """
    Preprocess the dataset to extract training rows, add the model's predictions
    and convert to wide format.
    """
    dataset_df_copy = dataset_df.copy()

    # filter training split
    tr_df = dataset_df_copy[dataset_df_copy[f"{metric}_split"] == "train"]

    # filter relevant columns and convert metric column to a list.
    tr_df = tr_df[["ArticleID", "System", metric]]
    tr_df[metric] = tr_df[metric].apply(ast.literal_eval)

    return _to_wide(tr_df, preds_df, metric=metric)


def _to_wide(df, preds_df, metric) -> pd.DataFrame:
    """
    Converts a dataframe with one row per (article, system, labeler) to
    a dataframe with one row per (article, system) and columns for each
    labeler's label. Also adds the model's prediction for
    the article-system pair.
    """
    print(preds_df.columns)
    preds_df.rename(columns={"article_id": "ArticleID"}, inplace=True)
    return (
        df.explode(metric)
        .groupby(["ArticleID", "System"], group_keys=False)
        .apply(add_labeler_id_column)
        .pivot_table(
            index=["ArticleID", "System"], columns="labeler_id", values=metric
        )
        .reset_index()
        .merge(preds_df, how="inner", on=["ArticleID", "System"])
    )
