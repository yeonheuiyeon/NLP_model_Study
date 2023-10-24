import copy
import datasets
import itertools
import torch
import re
from transformers import LlamaTokenizer
from llama_recipes.datasets.utils import ConcatDataset
from llama_recipes.datasets.utils import Concatenator
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokenizer.add_special_tokens({"pad_token": "<PAD>",})

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
IGNORE_INDEX = -100
MAX_WORDS=2048

def fine_tune_data2(data_point):
    prompt=data_point['sorce']
    example = prompt + data_point["target"]
    prompt = tokenizer(prompt,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
    
    example = tokenizer(example,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
    padding = MAX_WORDS - example.shape[0]

    if padding > 0:
        example = torch.cat((example, torch.Tensor(padding, dtype=torch.int64) - 1).fill_(tokenizer.pad_token_id))
    elif padding < 0:
        example = example[: MAX_WORDS]
    labels = copy.deepcopy(example)
    labels[: len(prompt)] = -1
    #example_mask = example.ne(tokenizer.pad_token_id)
    label_mask = labels.ge(0)
    #example[~example_mask] = 0
    labels[~label_mask] = IGNORE_INDEX
    #example_mask = example_mask.float()
    #label_mask = label_mask.float()
    return {
        "input_ids": example,
        "labels": labels,

    }


def tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = tokenizer(
            strings,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

    input_ids = labels = tokenized_list['input_ids']
    input_ids_lens = labels_lens = [
        tokenized.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in input_ids
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens
    )

def preprocess(
        sources,
        targets,
        tokenizer):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        tokenize_fn(strings, tokenizer) for strings in [examples, sources]
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    attention_masks=[j.ne(tokenizer.pad_token_id).int() for j in input_ids]
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    data_all=[ {'input_ids':i,'labels':l}  for i,l,a in zip(input_ids, labels,attention_masks)]
    print(data_all[0], len(data_all[0]['input_ids']),len(data_all[0]['labels']))
    for d in data_all:
        assert len(d['input_ids'])==len(d['labels']), f"{len(d['input_ids'])},{len(d['labels'])}"
    return datasets.Dataset.from_list(data_all)



def fine_tune_data(data_point):
    sample = f'### Instruction:\nProvide an appropriate answer to the question and support your response with accurate evidence. Question: {data_point["Q"]}\n\n### Response: '
    texts=data_point["A"]+'\n\n'
    ref=data_point["Rationale"]
    for i in list(set(re.findall('\[(.)\]', texts))):
        try:
            ref_num = int(i)
            texts += f'[{ref_num}] : {ref[ref_num]}\n'
        except:
            continue

    #example = sample+texts

    return {
        "sorce":sample,
        "target":texts,

    }


def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("THUDM/webglm-qa", split="train")

    dataset = dataset.map(lambda sample: {
        "Q": sample["question"],
        "A": sample["answer"],
        "Rationale": sample["references"],
        },
        batched=True,
        remove_columns=list(dataset.features),)

    dataset=dataset.map(fine_tune_data,remove_columns=['Q','A','Rationale'])
    #data_dict = preprocess(dataset['sorce'], dataset['target'], tokenizer)
    dataset = dataset.map(fine_tune_data2, remove_columns=['sorce', 'target'])
    for idx,i in enumerate(dataset):
        if idx>2:
            break
        print(i)
    return dataset
