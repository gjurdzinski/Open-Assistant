import pandas as pd
import ast
from pathlib import Path
from typing import Dict, List, Literal

Metric = Literal["InformativenessRating", "RelevanceRating"]
Split = Literal["train", "valid"]

_INCREMENTAL_NAMES_PARTIAL = [
    "rewards_00",
    "rewards_10",
]
_INCREMENTAL_NAMES = [
    "rewards_00",
    "rewards_01",
    "rewards_02",
    "rewards_03",
    "rewards_04",
    "rewards_05",
    "rewards_06",
    "rewards_07",
    "rewards_08",
    "rewards_09",
    "rewards_10",
]
_INCREMENTAL_NAMES_BUCKETS_PARTIAL = [
    f"buckets/{name}" for name in _INCREMENTAL_NAMES_PARTIAL
]
METRIC_2_MODEL_INCREMENTAL_PARTIAL = {
    "InformativenessRating": _INCREMENTAL_NAMES_PARTIAL,
}
#     "RelevanceRating": _INCREMENTAL_NAMES,
# }
METRIC_2_MODEL_INCREMENTAL_BUCKETS_PARTIAL = {
    "InformativenessRating": _INCREMENTAL_NAMES_BUCKETS_PARTIAL,
}
#     "RelevanceRating": _INCREMENTAL_NAMES_BUCKETS,
# }
_INCREMENTAL_NAMES_BUCKETS = [f"buckets/{name}" for name in _INCREMENTAL_NAMES]
METRIC_2_MODEL_INCREMENTAL = {
    "InformativenessRating": _INCREMENTAL_NAMES,
}
#     "RelevanceRating": _INCREMENTAL_NAMES,
# }
METRIC_2_MODEL_INCREMENTAL_BUCKETS = {
    "InformativenessRating": _INCREMENTAL_NAMES_BUCKETS,
}
#     "RelevanceRating": _INCREMENTAL_NAMES_BUCKETS,
# }
METRIC_2_MODEL = {
    "InformativenessRating": [
        "deberta-v3-small-4096-reward",
        "open-assistant-deberta-v3-base-3072-reward",
        "deberta-v3-base-1250-median",
        "deberta-v3-small-1750-median",
        "deberta-v3-small-4096-median",
    ],
    "RelevanceRating": [
        "deberta-v3-small-4096-reward",
        "open-assistant-deberta-v3-base-3072-reward",
        "deberta-v3-base-1250-median",
        "deberta-v3-small-1750-median",
        "deberta-v3-small-4096-median",
    ],
}


def load_models_predictions(
    metric2model: Dict[str, List[str]] = METRIC_2_MODEL,
    preds_path: str = "~/datasets/newsroom/incremental",
    load_from_buckets_dir: bool = False,
    split_prefixes: List[str] = ["tr_", "val_"],
    verbose: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Loads and processes the predictions of the models.
    Args:
        metric2model: dict mapping metric names to model names.
        preds_path: path to the predictions.
        load_from_buckets_dir: whether to load the predictions from
            the buckets directory.
        split_prefixes: prefixes of the splits to be read.
        verbose: whether to print progress.
    """
    path = Path(preds_path)
    metric2model2output = {}
    for metric in metric2model.keys():
        metric2model2output[metric] = {}
    for metric, models in metric2model.items():
        for model in models:
            for prefix in split_prefixes:
                split_model = f"{prefix}{model}"
                csv_file = (
                    path / f"{metric}/buckets/{split_model}.csv"
                    if load_from_buckets_dir
                    else path / f"{metric}/{split_model}.csv"
                )
                if verbose:
                    print(
                        f"Loading predictions for {split_model} on {metric} "
                        f"from {csv_file}..."
                    )
                metric2model2output[metric][split_model] = pd.read_csv(
                    csv_file
                )
    return metric2model2output


def load_models_predictions_new(
    preds_path: str,
    model: str,
    run: str,
    metric: str,
    percentage: str,
    filename: str,
    split: str,
    load_from_buckets_dir: bool = False,
) -> pd.DataFrame:
    path = Path(preds_path) / model / run / metric / percentage
    preds_csv = (
        path / "buckets" / filename
        if load_from_buckets_dir
        else path / filename
    )
    df = pd.read_csv(preds_csv)
    df["model"] = model
    df["run"] = run
    df["metric"] = metric
    df["percentage"] = percentage
    df["split"] = split
    return df


def load_labelers_predictions(
    dataset_dir: str,
    filename: str,
    metric: Metric,
    split: Split = "valid",
    exclude_abstractive: bool = True,
) -> pd.DataFrame:
    """
    Loads and processes the predictions of the labelers.
    Args:
        dataset_dir: path to the dataset.
        filename: name of the file containing the predictions.
        metric: name of the metric.
        exclude_abstractive: whether to exclude the abstractive system.
    """
    df = pd.read_csv(Path(dataset_dir) / filename)
    df = df[
        [
            "ArticleID",
            "System",
            "ArticleText",
            "SystemSummary",
            "ArticleTitle",
            metric,
            f"{metric}_split",
        ]
    ]

    df[metric] = df[metric].apply(ast.literal_eval)

    df = df[df[f"{metric}_split"] == split]
    # df = df.query(f"`{metric}_split` == `{split}`")
    if exclude_abstractive:
        df = df.query("`System` != 'abstractive'")

    df = df.explode(metric)

    df = df.groupby(["ArticleID", "System"], group_keys=False).apply(
        add_labeler_id_column
    )

    df = (
        df[["ArticleID", "System", "labeler_id", metric]]
        .pivot_table(
            index=["ArticleID", "System"], columns="labeler_id", values=metric
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .assign(
            labeler_mean=lambda x: (
                x["labeler_1"] + x["labeler_2"] + x["labeler_3"]
            )
            / 3.0,
            summary_id=lambda x: x["ArticleID"].astype(str)
            + "_"
            + x["System"],
        )
        .sort_values("labeler_mean")
    )

    return df


def add_labeler_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["labeler_id"] = [
        f"labeler_{i}" for i in range(1, len(df_copy) + 1, 1)
    ]
    return df_copy
