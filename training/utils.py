import re
from typing import AnyStr, List

import yaml
from sklearn.model_selection import train_test_split
from tokenizers import pre_tokenizers
from torch.utils.data import Subset
from transformers import AutoTokenizer

# @agoryuno contributed this
re_reference_remove = re.compile(r"\[\d+(?:,\s*\d+)*?\]")


def get_tokenizer(tokenizer_name, max_length, per_digit_tokens=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, truncation=True, max_length=max_length)

    if per_digit_tokens:
        tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    return tokenizer


def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=val_split,
        random_state=666,
        shuffle=True,
    )
    # [3879, 11479, 8341, 9177, 10798, 18177, 5735, 15669, 4837, 2760]
    print(val_idx[:10])
    # [13582, 5919, 11875, 7373, 19135, 13706, 8555, 15788, 15005, 15209]
    print(train_idx[:10])
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def freeze_top_n_layers(model, target_layers):
    # its possible we can simply detect which module is a ModuleList
    # and simply freeze the module without doing string parsing
    for name, param in model.named_parameters():
        if "embed" in name:
            param.requires_grad = False
        elif ".layer" in name or ".h." in name:
            tokens = name.split(".")
            idx = 0
            for token in tokens:
                if "layer" in token or token == "h":
                    break
                idx += 1
            if idx >= len(tokens):
                continue

            layer_ = int(tokens[idx + 1])
            if layer_ < target_layers:
                # print('freeze ', layer_, name)
                param.requires_grad = False
    return model


def argument_parsing(parser):
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        training_conf = yaml.safe_load(f.read())

    default_params = {
        "train_splits": [["train"]],
        "num_train_epochs": 4,
        "learning_rate": 3e-5,
        "eval_steps": 500,
        "loss": "rank",
        "warmup_steps": 500,
        "max_length": 440,
        "weight_decay": 0.01,
        "max_grad_norm": 2.0,
        "save_steps": 500,
        "per_device_eval_batch_size": 5,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": False,
        "local_rank": -1,
        "datasets": ["webgpt"],
        "wandb_entity": "open-assistant",
        "per_digit_tokens": False,
        "fp16": True,
        "tokenizer_name": training_conf["model_name"],
        "output_dirs": ["output"],
        "auto_find_batch_size": False,
        "report_to": [],
    }
    args_without_none = {k: v for (k, v) in vars(args).items() if v is not None}
    if not args_without_none["per_digit_tokens"]:  # Don't let missing command line override the conf
        del args_without_none["per_digit_tokens"]

    # Apply default params, then yaml config, then command line args where specific (i.e. not None)
    params = {**default_params, **training_conf, **args_without_none}
    for name in [
        "gradient_accumulation_steps",
        "num_train_epochs",
        "save_steps",
        "eval_steps",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
    ]:
        params[name] = int(params[name])
    for name in ["learning_rate", "weight_decay", "max_grad_norm"]:
        params[name] = float(params[name])

    print("params", params)

    return params


def get_datasets(dataset_list: List[AnyStr], summeval_path=None, train_splits=[]):
    from rank_datasets import NewsroomDataset, SummevalDataset
    from torch.utils.data import ConcatDataset

    train_datasets, evals = [], {}
    for dataset_name in dataset_list:
        if "summeval_local" == dataset_name and summeval_path is not None:
            train = SummevalDataset(dataset_path=summeval_path, splits=train_splits)
            train_datasets.append(train)
            eval = SummevalDataset(dataset_path=summeval_path, splits=["validation"])
            evals["summeval_local"] = eval
        elif "newsroom_local" == dataset_name and summeval_path is not None:
            train = NewsroomDataset(dataset_path=summeval_path, splits=train_splits)
            print(" >>>> read data from:", summeval_path)
            train_datasets.append(train)
            eval = NewsroomDataset(dataset_path=summeval_path, splits=["valid"])
            evals["newsroom_local"] = eval

    train = ConcatDataset(train_datasets)
    return train, evals
