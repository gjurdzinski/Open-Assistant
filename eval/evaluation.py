from typing import Literal
import pandas as pd

NormalizationTypes = Literal[None, "mean", "mean_std"]


def evaluate_labelers(
    df: pd.DataFrame,
    normalization: NormalizationTypes = None,
    human_prefix: str = "crowdsourced_",
    include_annotator: bool = False,
    k: int = 1,
):
    """
    Args:
      df: DataFrame where each row is a one example and each column is
        an annotator. The values should already be normalized to the [0,1]
        range.
      normalization: string denoting the normalization to be applied to each
        column:
        * None - no normalization
        * 'mean' - subtract mean
        * 'mean_std' - substract mean and divide by standard deviation
      human_prefix: prefix of the columns' names that store the reference
        annotations.
      include_annotator: whether to lables of evaluated annotators in the mean
        calculation at the example level.
      k: exponentiation of absolute value of differences.
    """
    df = df.copy()
    crowdsourced_annotators = [
        col_name
        for col_name in df.columns
        if str(col_name).startswith(human_prefix)
    ]
    other_annotators = [
        col_name
        for col_name in df.columns
        if col_name not in crowdsourced_annotators
    ]

    means = df.mean(axis=0)
    stds = df.std(axis=0)

    if normalization and "mean" in normalization:
        df = df - means
        if "std" in normalization:
            df = df / stds

    crowdsourced_s_dash = df[crowdsourced_annotators].mean(axis=1)
    differences_df = df.sub(crowdsourced_s_dash, axis=0)
    abs_differences_df = differences_df.abs().pow(k)

    out = 1 - (abs_differences_df.sum(axis=0) / len(abs_differences_df))
    # out = (abs_differences_df.sum(axis=0) / len(abs_differences_df))
    return out
