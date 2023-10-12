# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools
import torch
from transformers import LlamaTokenizer
from llama_recipes.datasets.utils import ConcatDataset
from llama_recipes.datasets.utils import Concatenator
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokenizer.add_special_tokens({"pad_token": "<PAD>",})
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
IGNORE_INDEX = -100
MAX_WORDS=350
def fine_tune_data(data_point):

    prompt=f'{B_INST} {data_point["Q"]} {E_INST}'
    #print(prompt)
    example = prompt + data_point["A"]
    #print(example)
    prompt = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.int64
    )
    example = tokenizer.encode(example)
    example.append(tokenizer.eos_token_id)
    example = torch.tensor(
        example, dtype=torch.int64
    )
    padding = MAX_WORDS - example.shape[0]

    if padding > 0:
        example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        example = example[: MAX_WORDS]
    labels = copy.deepcopy(example)
    labels[: len(prompt)] = -1
    example_mask = example.ge(0)
    label_mask = labels.ge(0)
    example[~example_mask] = 0
    labels[~label_mask] = IGNORE_INDEX
    example_mask = example_mask.float()
    label_mask = label_mask.float()
    return {
        "input_ids": example,
        "labels": labels,
        "attention_mask": example_mask,
    }


def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("Open-Orca/OpenOrca", split="train")

    dataset = dataset.map(lambda sample: {
        "Q": sample["question"],
        "A": sample["response"],
        },
        batched=True,
        remove_columns=list(dataset.features),)

    dataset=dataset.map(fine_tune_data,remove_columns=['Q','A'])
    for idx,d in enumerate(dataset):
        if idx>3:
            break
        print(d)

   
    dataset = dataset.map(Concatenator(), batched=True)
    for idx,d in enumerate(dataset):
        if idx>3:
            break
        print(d)

    return dataset

