import glob
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from rank_datasets import DataCollatorForPairRank
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames
from utils import (
    argument_parsing,
    freeze_top_n_layers,
    get_datasets,
    get_tokenizer,
)

accuracy = evaluate.load("accuracy")
parser = ArgumentParser()
# Note, these override the config yaml, and get merged in argument_parsing() in utils.py
# Do not set defaults here, but set them in utils.py so that the config yaml can override them.
parser.add_argument("config", type=str)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--per-digit-tokens", action="store_true")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(
        predictions=predictions, references=[0] * predictions.shape[0]
    )


class RankLoss(nn.Module):
    def __init__(self, eps=1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, pos, neg):
        loss = -self.log_sigmoid(pos - neg + self.eps).mean()
        return loss


class RankTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        model_name: str = None,
        args: Optional[TrainingArguments] = None,
        loss_function: str = "rank",
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.loss_fct = (
            RankLoss() if loss_function == "rank" else nn.CrossEntropyLoss()
        )
        self.loss_function = loss_function
        self.model_name = model_name

    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        if "rankgen" in self.model_name:
            positive_outputs = model(inputs["prefix"], inputs["positive"])
            negative_outputs = model(inputs["prefix"], inputs["negative"])
            if self.loss_function == "rank":
                loss = self.loss_fct(positive_outputs, negative_outputs)
            else:
                raise NotImplementedError(
                    "Only ranking loss has been implemented for rankgen model"
                )
            outputs = torch.hstack(
                (positive_outputs[:, None], negative_outputs[:, None])
            )
        else:
            inputs.pop("token_type_ids", None)
            outputs = model(**inputs)
            logits = outputs.get("logits").view(-1, 2)
            if self.loss_function == "rank":
                loss = self.loss_fct(logits[:, 0], logits[:, 1])
            else:
                loss = self.loss_fct(
                    logits,
                    torch.zeros(
                        logits.shape[0], device=logits.device, dtype=torch.long
                    ),
                )

        return (loss, outputs) if return_outputs else loss

    def _compute_loss(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        logits = outputs.get("logits").view(-1, 2)
        if self.loss_function == "rank":
            loss = self.loss_fct(logits[:, 0], logits[:, 1])
        else:
            loss = self.loss_fct(
                logits,
                torch.zeros(
                    logits.shape[0], device=logits.device, dtype=torch.long
                ),
            )

        return loss, logits

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        with torch.inference_mode():
            if "rankgen" in self.model_name:
                inputs = self._prepare_inputs(inputs)
                positive_outputs = model(inputs["prefix"], inputs["positive"])
                negative_outputs = model(inputs["prefix"], inputs["negative"])
                if self.loss_function == "rank":
                    loss = self.loss_fct(positive_outputs, negative_outputs)
                else:
                    raise NotImplementedError(
                        "Only ranking loss has been implemented for rankgen model"
                    )
                logits = torch.hstack(
                    (positive_outputs[:, None], negative_outputs[:, None])
                )
                # Create labels which are not None so HF will call compute_metrics:
                labels = torch.zeros(
                    logits.shape[0], device=logits.device, dtype=torch.long
                )
                return loss, logits, labels
            else:
                loss, logits = self._compute_loss(model, inputs)

                loss = loss.mean().detach()
                labels = torch.zeros(
                    logits.shape[0], device=logits.device, dtype=torch.long
                )
                if self.args.prediction_loss_only:
                    return loss, None, None

                return loss, logits, labels


def train_procedure(training_conf, iteration):
    """Train a model on a given set of train datasets."""
    training_conf = argument_parsing(parser)

    model_name = training_conf["model_name"]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    )
    if "freeze_layer" in training_conf:
        num_layer = training_conf["freeze_layer"]
        model = freeze_top_n_layers(model, num_layer)
        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters()
        )
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of trainable : {}M".format(int(params / 1e6)))

    optimizer = OptimizerNames.ADAMW_HF
    output_dir = training_conf["output_dirs"][iteration]
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_conf["num_train_epochs"],
        warmup_steps=training_conf["warmup_steps"],
        optim=optimizer,
        lr_scheduler_type=training_conf["scheduler"],
        learning_rate=training_conf["learning_rate"],
        # half_precision_backend="apex",
        fp16=training_conf["fp16"],
        local_rank=training_conf["local_rank"],
        gradient_checkpointing=training_conf["gradient_checkpointing"],
        gradient_accumulation_steps=training_conf[
            "gradient_accumulation_steps"
        ],
        per_device_train_batch_size=training_conf[
            "per_device_train_batch_size"
        ],
        per_device_eval_batch_size=training_conf["per_device_eval_batch_size"],
        weight_decay=training_conf["weight_decay"],
        max_grad_norm=training_conf["max_grad_norm"],
        logging_steps=10,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=training_conf["eval_steps"],
        save_steps=training_conf["save_steps"],
        auto_find_batch_size=training_conf["auto_find_batch_size"],
        load_best_model_at_end=True,
        report_to=training_conf["report_to"],
        overwrite_output_dir=True,
        metric_for_best_model="accuracy",
    )

    tokenizer = get_tokenizer(
        training_conf["tokenizer_name"],
        training_conf["max_length"],
        per_digit_tokens=training_conf["per_digit_tokens"],
    )
    train, evals = get_datasets(
        training_conf["datasets"],
        **{
            "summeval_path": value
            for key, value in training_conf.items()
            if key == "summeval_path"
        },
        **{
            "train_splits": value[iteration]
            for key, value in training_conf.items()
            if key == "train_splits"
        },
    )

    collate_fn = DataCollatorForPairRank(
        tokenizer,
        max_length=training_conf["max_length"],
        drop_token_type=training_conf.get("drop_token_type", False),
    )
    assert len(evals) > 0

    if "wandb" in training_conf["report_to"]:
        import wandb

        run = wandb.init(
            project="gpt-novel-multi",
            name=f"shard_{iteration}",
            group=training_conf["summeval_path"],
            reinit=True,
        )

    trainer = RankTrainer(
        model=model,
        model_name=model_name,
        args=args,
        loss_function=training_conf["loss"],
        train_dataset=train,
        eval_dataset=torch.utils.data.ConcatDataset(evals.values()),
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # optimizers=(optimizer, scheduler),
    )

    # evaluate at step zero:
    trainer.evaluate()

    # This is a hack to allow empty list of train datasets to easily skip
    # training and evaluete non-fine-tuned model.
    if train.cumulative_sizes[0] > 0:
        trainer.train()
        trainer.evaluate()

    # save the best model:
    best_model_path = Path(output_dir) / "checkpoint-best"
    trainer.save_model(best_model_path)

    print("final_inference...")

    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path
    ).to(device)

    dataset_dict = DatasetDict.load_from_disk(training_conf["summeval_path"])

    get_rewards_and_save(
        tokenizer,
        model,
        dataset_dict["train_final"],
        output_dir,
        "tr_rewards.csv",
    )

    get_rewards_and_save(
        tokenizer,
        model,
        dataset_dict["valid_final"],
        output_dir,
        "val_rewards.csv",
    )

    if "wandb" in training_conf["report_to"]:
        run.finish()

    # remove all checkpoints
    pattern = str(Path(output_dir) / "checkpoint*")
    matching_dirs = glob.glob(pattern)
    for dir_path in matching_dirs:
        shutil.rmtree(dir_path)


def get_rewards_and_save(tokenizer, model, final_ds, output_dir, filename):
    def tokenize(batch):
        return tokenizer(
            batch["deberta_input"],
            padding=True,
            truncation=True,
            max_length=training_conf["max_length"],
        )

    final_encoded = final_ds.map(
        tokenize,
        batched=True,
        batch_size=training_conf["per_device_eval_batch_size"],
    )
    final_encoded.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask"]
    )
    final_dataloader = DataLoader(
        final_encoded,
        batch_size=training_conf["per_device_eval_batch_size"],
        shuffle=False,
    )

    outputs_buffer = []

    for batch in tqdm(final_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        outputs_buffer.append(outputs.logits[:, 0])

    summary_scores = torch.cat(outputs_buffer, dim=0).cpu().numpy()

    rewards_df = pd.DataFrame(
        data={
            "ArticleID": final_ds["ArticleID"],
            "System": final_ds["System"],
            "deberta_reward": summary_scores,
        }
    )

    rewards_df.to_csv(Path(output_dir) / filename, index=False)


if __name__ == "__main__":
    training_conf = argument_parsing(parser)

    assert len(training_conf["train_splits"]) == len(
        training_conf["output_dirs"]
    )
    for iteration, _config in enumerate(training_conf["train_splits"]):
        print(f" >> {iteration} | {_config}")
        train_procedure(training_conf, iteration)
