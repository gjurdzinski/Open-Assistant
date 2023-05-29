"""
TODO: refactor to take arguments from command line instead of hardcoding
values.

TODO: The filenames hardcoded are a complete mess and repetition (see
utils.py). Fix it.
"""

import pandas as pd
from pathlib import Path
from eval.eval_utils import (
    load_models_predictions,
    METRIC_2_MODEL_INCREMENTAL_PARTIAL,
)
from bucketing import get_buckets_bounds_and_scaler, map_to_optimal_buckets

HOME_DIR = "/mnt/ml-team/homes/grzegorz.jurdzinski"
NEWSROOM_DIR = f"{HOME_DIR}/datasets/newsroom"
# PREDS_DIR = f"{HOME_DIR}/runs/rewards/incremental/reward-deberta-large"
PREDS_DIR = f"{HOME_DIR}/runs/newsroom-incremental/deberta-base/B"
METRICS = ("InformativenessRating",)  # "RelevanceRating")
# INCREMENTAL_FILENAMES = [f"rewards_{i:02}" for i in range(11)]
INCREMENTAL_FILENAMES = ["rewards_01"]
DATASET_FILE = f"{NEWSROOM_DIR}/newsroom-aggregated-sorted-order.csv"

# d = {
#     "deberta-base": {
#         "A": {
#             "InformativenessRating": {
#                 "percentage": {
#                     "000": []
#                 }
#             }
#         }
#     }
# }

df = pd.read_csv(DATASET_FILE)

models_predictions = load_models_predictions(
    metric2model=METRIC_2_MODEL_INCREMENTAL_PARTIAL,
    preds_path=PREDS_DIR,
    load_from_buckets_dir=False,
    verbose=False,
)

for metric in METRICS:
    buckets_dir = Path(f"{PREDS_DIR}/{metric}/buckets")
    buckets_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving bucketised predictions to {buckets_dir}/...")

    for name in INCREMENTAL_FILENAMES:
        print(f"Processing {name}...")
        bounds, scaler = get_buckets_bounds_and_scaler(
            dataset_df=df,
            tr_preds_df=models_predictions[metric][f"tr_{name}"],
            metric=metric,
        )

        # bucketise and save the validation predictions
        bucketised = map_to_optimal_buckets(
            models_predictions[metric][f"val_{name}"],
            bounds,
            scaler,
        )
        bucketised.to_csv(
            f"{buckets_dir}/val_{name}.csv",
            index=False,
        )

        # bucketise and save the train predictions
        bucketised = map_to_optimal_buckets(
            models_predictions[metric][f"tr_{name}"],
            bounds,
            scaler,
        )
        bucketised.to_csv(
            f"{buckets_dir}/tr_{name}.csv",
            index=False,
        )
    print(30 * "#")
