"""
    author: theblackcat102

    Dataset output format from __getitem__

     - question / prompt : string

     - answers / rows : list of tuple pair. The first element in the tuple pair must be the positive pair (rank higher than the second element)

    A list of rank based dataset for training using rank loss

    Some nice features to have

    [] support additional negative samples generated from other models.

        For example we can use galactica-125m to generate a TLDR and assume it was
        inferior than the human preference one


"""
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from datasets import DatasetDict
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)


@dataclass
class DataCollatorForPairRank:
    """

    Data collator that will dynamically pad the inputs for multiple choice received.

    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    drop_token_type: bool = False  # galactica

    def __call__(self, features):
        flatten_features = []
        for question, pairs in features:
            for pos, neg in pairs:
                flatten_features.append(
                    self.tokenizer(
                        question,
                        pos,
                        truncation=True,
                        max_length=self.max_length,
                    )
                )
                flatten_features.append(
                    self.tokenizer(
                        question,
                        neg,
                        truncation=True,
                        max_length=self.max_length,
                    )
                )
        batch = self.tokenizer.pad(
            flatten_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if self.drop_token_type:
            batch.pop("token_type_ids")
        return batch


class NewsroomDataset(Dataset):
    def __init__(
        self, dataset_path, splits=["train"], max_comparison_per_sample=1
    ) -> None:
        super().__init__()
        self.summaries = {}
        # using prompt as our index will allows us
        # to add additional generated prompt later
        self.index2summary = {}
        self.max_comparison_per_sample = max_comparison_per_sample
        # major_split = split if "train" == split else "validation"
        dataset_dict = DatasetDict.load_from_disk(dataset_path)
        for split in splits:
            dataset = dataset_dict[split]
            summaries = {}
            for data in dataset:
                context = data["article"]

                if context not in self.index2summary:
                    self.index2summary[len(self.index2summary)] = context

                if context not in summaries:
                    summaries[context] = []

                pos, neg = (
                    ("first_summary", "second_summary")
                    if data["choice"] == 0
                    else ("second_summary", "first_summary")
                )
                summaries[context].append(
                    (
                        data[pos],
                        data[neg],
                    )
                )

            self.summaries = {**self.summaries, **summaries}

        self.postfix_prompt = " TLDR; "

    def __len__(self):
        return len(self.index2summary)

    def __getitem__(self, index):
        context = self.index2summary[index]
        # return pairs of comparison
        rows = self.summaries[context]
        # pair very big
        # we are going to do some sampling
        # not optimal but good for now
        valid_idx = np.random.choice(len(rows), self.max_comparison_per_sample)
        # optimize the format later
        return context + self.postfix_prompt, [
            r for idx, r in enumerate(rows) if idx in valid_idx
        ]


class SummevalDataset(Dataset):
    """
    labeling method : pair comparison, 0 or 1
    """

    def __init__(
        self, dataset_path, splits=["train"], max_comparison_per_sample=1
    ) -> None:
        super().__init__()
        self.summaries = {}
        # using prompt as our index will allows us
        # to add additional generated prompt later
        self.index2summary = {}
        self.max_comparison_per_sample = max_comparison_per_sample
        if type(split) == str:
            splits = [splits]
        dataset_dict = DatasetDict.load_from_disk(dataset_path)
        for split in splits:
            dataset = dataset_dict[split]
            summaries = {}
            for data in dataset:
                context = data["article"]

                if context not in self.index2summary:
                    self.index2summary[len(self.index2summary)] = context

                if context not in summaries:
                    summaries[context] = []

                pos, neg = (
                    ("first_summary", "second_summary")
                    if data["choice"] == 0
                    else ("second_summary", "first_summary")
                )
                summaries[context].append(
                    (
                        data[pos],
                        data[neg],
                    )
                )

            self.summaries = {**self.summaries, **summaries}

        self.postfix_prompt = " TLDR;"

    def __len__(self):
        return len(self.index2summary)

    def __getitem__(self, index):
        context = self.index2summary[index]
        # return pairs of comparison
        rows = self.summaries[context]
        # pair very big
        # we are going to do some sampling
        # not optimal but good for now
        valid_idx = np.random.choice(len(rows), self.max_comparison_per_sample)
        # optimize the format later
        return context + self.postfix_prompt, [
            r for idx, r in enumerate(rows) if idx in valid_idx
        ]
