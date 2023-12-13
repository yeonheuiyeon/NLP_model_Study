import torch
import pandas as pd
from torch.utils.data import Dataset
import utils
import transformers
import logging
import jsonlines
from typing import Sequence, Dict
import copy
import random
import pathlib
import os
import jsonlines
from dataclasses import dataclass, field
from datasets import Dataset,load_from_disk, load_dataset, concatenate_datasets

IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_w_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_wo_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_wo_input2": (
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "vicuna_base": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "\n\n"
        "{input}Assistant:"
    ),
    "q2q": (
        "Generate a keyword and a similar question of the following which maps to the same answer.\n"
        "Input: {input}\n"
        "Output:"
    ),
    "qa": (
        "Question: {instruction}\nAnswer:"
    ),
    "instruction": (
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "none": (
        "{instruction}"
    ),
    "mrcorca_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        logging.warning("Loading data...")
        list_data_dict = utils.load_data(data_path)

        logging.warning("Formatting inputs...")
        sources = [
            example["input"]
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]
        self.sources = sources
        self.targets = targets

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def _tokenize_fn(
            self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
    ) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(
            self,
            sources: Sequence[str],
            targets: Sequence[str],
            tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            self._tokenize_fn(strings, tokenizer) for strings in (examples, sources)
        ]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class EmptyDataset(InstructionDataset):
    def __init__(self):
        self.sources = []
        self.targets = []
        self.input_ids = []
        self.labels = []


class FinetuningDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, source_label, target_label,
                 system_label, frac=1.0):
        logging.warning("Loading data...")
        list_data_dict = utils.load_data(data_path)
        list_data_dict = list_data_dict.select(
            list(random.sample(range(len(list_data_dict)), int(len(list_data_dict) * frac))))

        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT[system_label]
        sources = [
            system_prompt.format(instruction=example[source_label])
            for example in list_data_dict
        ]
        targets = [
            f"{example[target_label]}{tokenizer.eos_token}" for example in list_data_dict
        ]
        self.sources = sources
        self.targets = targets

        print(sources[0], targets[0])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

class Specific_train_Dataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        SUBJECTS = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
        ]
        logging.warning("Loading data...")
        system_prompt = PROMPT_DICT["none"]
        sources = []
        targets = []
        num_to_letter = {"0": "A", "1": "B", "2": "C", "3": "D", "4": "E"}

        for sub in SUBJECTS:
            list_data_dict = load_dataset("cais/mmlu",sub,split='auxiliary_train')
            # list_data_dict = list_data_dict.select(list(random.sample(range(len(list_data_dict)), int(len(list_data_dict)*frac))))

            logging.warning("Formatting inputs...")

            for example in list_data_dict:
                try:
                    ans = num_to_letter[str(example["answer"])]
                    op1 = example["choices"][0]
                    op2 = example["choices"][1]
                    op3 = example["choices"][2]
                    op4 = example["choices"][3]
                    #ans = example["answer"]
                    ques = example["question"]
                    insruction_orca = f"Question: {ques}\nA. {op1}\nB. {op2}\nC. {op3}\nD. {op4}\nAnswer:"
                    source = system_prompt.format(instruction=insruction_orca)
                    target = ' '+ans.strip()
                except:
                    print(example), ": failed"
                    continue
                sources.append(source)
                targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

class HellaswagDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        logging.warning("Loading data...")
        list_data_dict = utils.load_data(data_path)

        logging.warning("Formatting inputs...")
        sources = [
            example["ctx"]
            for example in list_data_dict
        ]
        targets = [
            f"{example['endings'][int(example['label'])]}{tokenizer.eos_token}" for example in list_data_dict
        ]

        print(sources[0], targets[0])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class MSDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        logging.warning("Loading data...")
        list_data_dict = pd.read_csv(data_path)

        logging.warning("Formatting inputs...")
        #system_prompt = PROMPT_DICT["prompt_wo_input"]
        sources = []
        targets = []
        for idx, example in enumerate(list_data_dict['targets']):
            try:
                insruction_orca = list_data_dict['inputs'][idx]
                source = insruction_orca
                target = ' '+example.strip()+tokenizer.eos_token
            except:
                continue
            sources.append(source)
            targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        print(len(sources))
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class CorrectionDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        logging.warning("Loading data...")
        list_data_dict = pd.read_csv(data_path)

        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT["prompt_wo_input"]
        sources = []
        targets = []
        for idx, example in enumerate(list_data_dict['output']):
            try:
                insruction_orca = list_data_dict['input'][idx]
                source = system_prompt.format(instruction=insruction_orca)
                target = example
            except:
                continue
            sources.append(source)
            targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

class QAOrcaDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        logging.warning("Loading data...")
        list_data_dict = load_dataset(data_path, split='train')

        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT["mrcorca_input"]
        sources = []
        targets = []
        for example in list_data_dict:
            insruction_orca = example['system_prompt']
            inputs=example['question']
            source = system_prompt.format(instruction=insruction_orca,input=inputs)
            target = ' '+example['response']+tokenizer.eos_token
            sources.append(source)
            targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

class QAOrcajsonDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        logging.warning("Loading data...")

        system_prompt = PROMPT_DICT["mrcorca_input"]
        sources = []
        targets = []
        with jsonlines.open(data_path) as f:
            for line in f.iter():
                insruction_orca = line['system_prompt']
                inputs=line['question']
                source = system_prompt.format(instruction=insruction_orca,input=inputs)
                target = ' '+line['response']+tokenizer.eos_token
                sources.append(source)
                targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class UltraDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        logging.warning("Loading data...")
        list_data_dict = load_dataset(data_path, split='train_sft')
        list_data_dict = list_data_dict.select(
            list(random.sample(range(len(list_data_dict)), int(len(list_data_dict) * frac))))

        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT["qa"]
        sources = []
        targets = []
        for example in list_data_dict:
            try:
                all_chat = [chat['content'] for chat in example['messages']]
                insruction_orca = all_chat[0]
                source = system_prompt.format(instruction=insruction_orca)
                target = ' '+' '.join(all_chat[1:])+tokenizer.eos_token
            except:
                continue
            sources.append(source)
            targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class Specific_mmlu_Dataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        logging.warning("Loading data...")
        SUBJECTS = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
        ]

        # list_data_dict = list_data_dict.select(list(random.sample(range(len(list_data_dict)), int(len(list_data_dict)*frac))))

        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT["none"]
        sources = []
        targets = []
        targetdict={0:'A',1:"B",2:"C",3:"D"}
        for task in SUBJECTS:
            list_data_dict=load_dataset("cais/mmlu",f"{task}",split='auxiliary_train')
            for idx, line in enumerate(list_data_dict):
                try:
                    q=line['question']
                    a=targetdict[line['answer']]
                    op=line['choices']
                    t=f"Question: {q}\nA. {op[0]}\nB. {op[1]}\nC. {op[2]}\nD. {op[3]}\nAnswer:"
                    source = system_prompt.format(instruction=t)
                    target =' '+a.strip()
                except:
                    print(idx), ": failed"
                    continue
                sources.append(source)
                targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class SpecificDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        logging.warning("Loading data...")
        list_data_dict =load_dataset("arrow", data_files={'train': data_path},split="train")
        #list_data_dict = list_data_dict.select(list(random.sample(range(len(list_data_dict)), int(len(list_data_dict)*frac))))
        #list_data_dict=pd.read_csv("/workspace/home/heuiyeen/Models/share_kollm/data/instruction_dataset_ko_ver_0_0_0/train/exaone.csv")
        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT["none"]
        sources = []
        targets = []
        for example in list_data_dict:
            try:
                insruction_orca = example["context"].strip()
                source = system_prompt.format(instruction=insruction_orca)
                target =example["text"].strip()
            except:
                print(": failed")
                continue
            sources.append(source)
            targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

class SlimorcaDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        logging.warning("Loading data...")
        list_data_dict = utils.load_data(data_path)
        list_data_dict = list_data_dict.select(
            list(random.sample(range(len(list_data_dict)), int(len(list_data_dict) * frac))))

        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT["prompt_w_input"]
        sources = []
        targets = []
        for example in list_data_dict:
            try:
                insruction_orca = example['conversations'][0]['value']
                source = system_prompt.format(instruction=insruction_orca, input=example['conversations'][1]['value'])
                target = example['conversations'][2]['value']
            except:
                continue
            sources.append(source)
            targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class OrcaBestDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, frac=1.0):
        logging.warning("Loading data...")
        list_data_dict = utils.load_data(data_path)
        list_data_dict = list_data_dict.select(
            list(random.sample(range(len(list_data_dict)), int(len(list_data_dict) * frac))))

        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT["prompt_w_input"]
        sources = []
        targets = []
        for example in list_data_dict:
            random_idx = random.randrange(example["num_samples"])
            source = system_prompt.format(instruction=example["instruction"],
                                          input=example["cluster"]["samples"][random_idx]["input"])
            target = example['cluster']['samples'][random_idx]['output']
            sources.append(source)
            targets.append(target)
        self.sources = sources
        self.targets = targets

        print(sources[0], targets[0])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class ExpertqaDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        logging.warning("Loading data...")
        list_data_dict = utils.load_data(data_path)

        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT["qa"]
        sources = [
            system_prompt.format(instruction=example['question'])
            for example in list_data_dict
        ]
        targets = [
            f"{example['answers'][list(example['answers'].keys())[0]]['revised_answer_string']}" for example in
            list_data_dict
        ]
        self.sources = sources
        self.targets = targets

        print(sources[0], targets[0])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class VicunaforagentDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        logging.warning("Loading data...")
        sources = []
        targets = []
        for split_name in ['os', 'db', 'alfworld', 'kg', 'webshop', 'mind2web']:
            list_data_dict = utils.load_data(data_path, split=split_name)
            logging.warning("Formatting inputs...")
            system_prompt = PROMPT_DICT["vicuna_base"]
            for example in list_data_dict:
                instruction = ""
                inputs = ""
                for i, conv in enumerate(example["conversations"][:-1]):
                    if i % 2 == 0:
                        inputs += f"USER: {conv['value']}{tokenizer.eos_token}\n"
                    else:
                        inputs += f"ASSISTANT: {conv['value']}{tokenizer.eos_token}\n"
                sources.append(system_prompt.format(instruction=instruction, input=inputs))
                targets.append(example["conversations"][-1]["value"] + tokenizer.eos_token)
        print(sources[0], targets[0])
        self.sources = sources
        self.targets = targets

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class VicunaDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        logging.warning("Loading data...")
        if os.path.isdir(data_path):
            sources = []
            targets = []
            for f in os.listdir(data_path):
                print(f)
                list_data_dict = utils.load_data(os.path.join(data_path, f))
                logging.warning("Formatting inputs...")
                system_prompt = PROMPT_DICT["vicuna_base"]
                for example in list_data_dict:
                    instruction = ""
                    inputs = ""
                    for i, conv in enumerate(example["conversations"][:-1]):
                        if i % 2 == 0:
                            inputs += f"USER: {conv['value']}{tokenizer.eos_token}\n"
                        else:
                            inputs += f"ASSISTANT: {conv['value']}{tokenizer.eos_token}\n"
                    sources.append(system_prompt.format(instruction=instruction, input=inputs))
                    targets.append(example["conversations"][-1]["value"] + tokenizer.eos_token)
        else:
            list_data_dict = utils.load_data(data_path)
            logging.warning("Formatting inputs...")
            system_prompt = PROMPT_DICT["vicuna_base"]
            sources = []
            targets = []
            for example in list_data_dict:
                instruction = ""
                inputs = ""
                for i, conv in enumerate(example["conversations"][:-1]):
                    if i % 2 == 0:
                        inputs += f"USER: {conv['value']}{tokenizer.eos_token}\n"
                    else:
                        inputs += f"ASSISTANT: {conv['value']}{tokenizer.eos_token}\n"
                sources.append(system_prompt.format(instruction=instruction, input=inputs))
                targets.append(example["conversations"][-1]["value"] + tokenizer.eos_token)
        print(sources[0], targets[0])
        self.sources = sources
        self.targets = targets

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]



class Sharegpt_all_Dataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        sources = []
        targets = []
        logging.warning("Loading data...")
        system_prompt = PROMPT_DICT["qa"]
        with jsonlines.open(data_path) as f:
            for line in f.iter():
                try:
                    all_chat = [chat['value'] for chat in line['conversations']]
                    insruction_orca = all_chat[0]
                    source = system_prompt.format(instruction=insruction_orca)
                    target = ' '.join(all_chat[1:]) + tokenizer.eos_token
                except:
                    continue
                sources.append(source)
                targets.append(target)

        self.sources = sources
        self.targets = targets

        print(sources[-1], targets[-1])
        print(len(sources), len(targets))
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]



class VicunaforUltraDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        logging.warning("Loading data...")

        list_data_dict = load_dataset(data_path, split='train_sft')
        logging.warning("Formatting inputs...")
        system_prompt = PROMPT_DICT["vicuna_base"]
        sources = []
        targets = []
        for example in list_data_dict:
            instruction = ""
            inputs = ""
            for i, conv in enumerate(example["messages"][:-1]):
                if i % 2 == 0:
                    inputs += f"USER: {conv['content']}{tokenizer.eos_token}\n"
                else:
                    inputs += f"ASSISTANT: {conv['content']}{tokenizer.eos_token}\n"
            sources.append(system_prompt.format(instruction=instruction, input=inputs))
            targets.append(example["messages"][-1]["content"] + tokenizer.eos_token)
        print(sources[0], targets[0])
        self.sources = sources
        self.targets = targets

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


class AlpacaDataset(InstructionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        logging.warning("Loading data...")
        if os.path.isdir(data_path):
            sources = []
            targets = []
            for f in os.listdir(data_path):
                print(f)
                list_data_dict = utils.load_data(os.path.join(data_path, f))
                logging.warning("Formatting inputs...")
                prompt_input, prompt_no_input = (
                    PROMPT_DICT["prompt_w_input"],
                    PROMPT_DICT["prompt_wo_input"],
                )
                sources += [
                    prompt_input.format_map(example)
                    if example.get("input", "") != ""
                    else prompt_no_input.format_map(example)
                    for example in list_data_dict
                ]
                targets += [
                    f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
                ]
        else:
            list_data_dict = utils.load_data(data_path)

            logging.warning("Formatting inputs...")
            prompt_input, prompt_no_input = (
                PROMPT_DICT["prompt_w_input"],
                PROMPT_DICT["prompt_wo_input"],
            )
            sources = [
                prompt_input.format_map(example)
                if example.get("input", "") != ""
                else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [
                f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
            ]
        self.sources = sources
        self.targets = targets
        print(sources[0], targets[0])

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def merge_datasets(dataset_list, tokenizer):
    new_dataset = EmptyDataset()
    for dataset in dataset_list:
        new_dataset.sources += dataset.sources
        new_dataset.targets += dataset.targets

    zipped = list(zip(new_dataset.sources, new_dataset.targets))
    random.shuffle(zipped)
    new_dataset.sources, new_dataset.targets = zip(*zipped)
    print('##############################################')
    print(new_dataset.sources[0], new_dataset.targets[0])

    data_dict = new_dataset.preprocess(new_dataset.sources, new_dataset.targets, tokenizer)
    new_dataset.input_ids = data_dict["input_ids"]
    new_dataset.labels = data_dict["labels"]

    return new_dataset


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if data_args.data_class in ["vicuna", "puffin"]:
        train_dataset = VicunaDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
        eval_dataset = None
    #        eval_dataset = VicunaDataset(
    #            tokenizer=tokenizer, data_path=data_args.eval_data_path
    #        )
    elif data_args.data_class in ["alpaca", "platypus"]:
        train_dataset = AlpacaDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
        eval_dataset = AlpacaDataset(
            tokenizer=tokenizer, data_path=data_args.eval_data_path
        )
    elif data_args.data_class == "correction":
        train_dataset = CorrectionDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
        eval_dataset = CorrectionDataset(
            tokenizer=tokenizer, data_path=data_args.eval_data_path
        )
    elif data_args.data_class == "hellaswag":
        train_dataset = HellaswagDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
    elif data_args.data_class == "flan-hellaswag":
        train_dataset = FinetuningDataset(
            tokenizer, data_args.train_data_path, "input", "target", "none"
        )
    elif data_args.data_class == "orca-best":
        train_dataset = OrcaBestDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
        eval_dataset = OrcaBestDataset(
            tokenizer=tokenizer, data_path=data_args.eval_data_path
        )
    elif data_args.data_class =="mmlu_cheat":
        train_dataset=Specific_mmlu_Dataset(
            tokenizer=tokenizer,data_path="")
        eval_dataset=Specific_mmlu_Dataset(
            tokenizer=tokenizer,data_path="")
    elif data_args.data_class == "Slimorca":
        train_dataset = SlimorcaDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
        eval_dataset = SlimorcaDataset(
            tokenizer=tokenizer, data_path=data_args.eval_data_path
        )
    elif data_args.data_class == "Mrcorca":
        train_dataset = QAOrcaDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
        eval_dataset = QAOrcaDataset(
            tokenizer=tokenizer, data_path=data_args.eval_data_path
        )
    elif data_args.data_class == "Ultra":
        train_dataset = UltraDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
        eval_dataset = UltraDataset(
            tokenizer=tokenizer, data_path=data_args.eval_data_path
        )
    elif data_args.data_class == "specificqa":
        train_dataset = SpecificDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
        eval_dataset = None
    elif data_args.data_class == "qaorca_json":
        train_dataset = QAOrcajsonDataset(
            tokenizer=tokenizer, data_path=data_args.train_data_path
        )
        eval_dataset = None
    elif data_args.data_class == "q2q":
        train_dataset = FinetuningDataset(
            tokenizer, data_args.train_data_path, "input", "output", "q2q"
        )
        eval_dataset = FinetuningDataset(
            tokenizer, data_args.train_data_path, "input", "output", "q2q"
        )
    elif data_args.data_class == "tiny-codes":
        train_dataset = FinetuningDataset(
            tokenizer, data_args.train_data_path, 'prompt', 'response', "none"
        )
        eval_dataset = FinetuningDataset(
            tokenizer, data_args.train_data_path, 'prompt', 'response', "none"
        )
    elif data_args.data_class == "curious":
        train_dataset = FinetuningDataset(
            tokenizer, data_args.train_data_path, 'question', 'answer', "qa"
        )
        eval_dataset = FinetuningDataset(
            tokenizer, ata_args.train_data_path, 'question', 'answer', "qa"
        )

    elif data_args.data_class == "math-instruct":
        train_dataset = FinetuningDataset(
            tokenizer, data_args.train_data_path, 'instruction', 'output', 'qa'
        )
        eval_dataset = FinetuningDataset(
            tokenizer, data_args.train_data_path, 'instruction', 'output', 'qa'
        )
    elif data_args.data_class == "comb1":
        sharegpt_dataset = VicunaDataset(tokenizer,
                                         "/workspace/home/data/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split_65002.jsonl")
        orcabest_dataset = OrcaBestDataset(tokenizer, 'shahules786/orca-best')
        train_dataset = merge_datasets([sharegpt_dataset, orcabest_dataset], tokenizer)
        eval_dataset = None
    elif data_args.data_class == "comb2":
        orcabest_dataset = OrcaBestDataset(tokenizer, 'shahules786/orca-best', 0.2)
        sharegpt_dataset = VicunaDataset(tokenizer,
                                         "/workspace/home/data/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split_65002.jsonl")
        train_dataset = merge_datasets([sharegpt_dataset, orcabest_dataset], tokenizer)
        eval_dataset = None
    elif data_args.data_class == "mmlu_cheat":
        sharegpt_dataset = SpecificDataset(tokenizer,"/workspace/home/heuiyeen/Code/exaone-post-train/data/target_mmluv2_analy.csv")
        train_dataset =sharegpt_dataset
        eval_dataset = None
    elif data_args.data_class == "mrcorca_ultra":
        qaorca_dataset = QAOrcaDataset(tokenizer, 'beaugogh/openorca-multiplechoice-10k')
        ultra_dataset = UltraDataset(tokenizer, 'HuggingFaceH4/ultrachat_200k')

        train_dataset = merge_datasets([qaorca_dataset, ultra_dataset], tokenizer)
        eval_dataset = None
    elif data_args.data_class == "comb_conv":
        puffin_dataset = VicunaDataset(tokenizer, 'LDJnr/Puffin')
        sharegpt_dataset = VicunaDataset(tokenizer, "./data/ShareGPT_V3_unfiltered_cleaned_split_65002.jsonl")
        ultrachat_data = VicunaforUltraDataset(tokenizer, 'HuggingFaceH4/ultrachat_200k')
        agentInstruct_data = VicunaforagentDataset(tokenizer, 'THUDM/AgentInstruct')
        train_dataset = merge_datasets([puffin_dataset, sharegpt_dataset, ultrachat_data, agentInstruct_data],
                                       tokenizer)
        eval_dataset = None
    elif data_args.data_class == "orcamix":
        qaorca_dataset = QAOrcaDataset(tokenizer, 'beaugogh/openorca-multiplechoice-10k')
        sharegpt_dataset = SlimorcaDataset(tokenizer, "Open-Orca/SlimOrca")
        train_dataset = merge_datasets([qaorca_dataset,sharegpt_dataset],
                                       tokenizer)
        eval_dataset = None
    elif data_args.data_class == "mrcorca_specific":
        #qaorca_dataset = QAOrcaDataset(tokenizer, 'beaugogh/openorca-multiplechoice-10k')
        all_science = MSDataset(tokenizer, "./data/alls_science_qa.csv")
        all_math = MSDataset(tokenizer, "./data/alls_math.csv")
        train_dataset = merge_datasets([all_math,all_science],
                                       tokenizer)
        eval_dataset = None
    elif data_args.data_class == "mrcorca_myversion":
        qaorca_dataset1 = QAOrcaDataset(tokenizer, 'beaugogh/openorca-multiplechoice-10k')
        qaorca_dataset2 = QAOrcajsonDataset(tokenizer, './data/train_mrcorca3.jsonl')
        train_dataset = merge_datasets([qaorca_dataset1,qaorca_dataset2],
                                       tokenizer)
        eval_dataset = None
    elif data_args.data_class == "mysubset_mrsubset":
        qaorca_dataset1 = QAOrcajsonDataset(
            tokenizer=tokenizer, data_path='./data/train_mrcorca5_subset.jsonl'
        )
        qaorca_dataset2 = QAOrcaDataset(tokenizer, 'beaugogh/openorca-multiplechoice-10k')
        train_dataset = merge_datasets([qaorca_dataset1,qaorca_dataset2],
                                       tokenizer)
        eval_dataset = None
    elif data_args.data_class == "comb3":
        sharegpt_dataset = VicunaDataset(tokenizer,
                                         "/workspace/home/data/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split_65002.jsonl")
        puffin_dataset = VicunaDataset(tokenizer, 'LDJnr/Puffin')
        platypus_dataset = AlpacaDataset(tokenizer, 'garage-bAInd/Open-Platypus')
        tinycode_dataset = FinetuningDataset(tokenizer, 'nampdn-ai/tiny-codes', 'prompt', 'response', "none", 0.01)
        curious_dataset = FinetuningDataset(tokenizer, 'xiyuez/im-feeling-curious', 'question', 'answer', "qa")
        mathinstruct_dataset = FinetuningDataset(tokenizer, 'TIGER-Lab/MathInstruct', 'instruction', 'output', 'qa',
                                                 0.1)
        orcabest_dataset = OrcaBestDataset(tokenizer, 'shahules786/orca-best', 0.1)
        expertqa_dataset = ExpertqaDataset(tokenizer, '/workspace/home/data/raw/expertqa_r2_compiled_anon.jsonl')
        train_dataset = merge_datasets(
            [sharegpt_dataset, puffin_dataset, platypus_dataset, tinycode_dataset, curious_dataset,
             mathinstruct_dataset, orcabest_dataset, expertqa_dataset], tokenizer)
        eval_dataset = None
    elif data_args.data_class == "comb4":
        sharegpt_dataset = VicunaDataset(tokenizer,
                                         "/workspace/home/data/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split_65002.jsonl")
        tinycode_dataset = FinetuningDataset(tokenizer, 'nampdn-ai/tiny-codes', 'prompt', 'response', "none", 0.01)
        curious_dataset = FinetuningDataset(tokenizer, 'xiyuez/im-feeling-curious', 'question', 'answer', "qa")
        mathinstruct_dataset = FinetuningDataset(tokenizer, 'TIGER-Lab/MathInstruct', 'instruction', 'output', 'qa',
                                                 0.1)
        train_dataset = merge_datasets(
            [sharegpt_dataset, puffin_dataset, platypus_dataset, tinycode_dataset, curious_dataset,
             mathinstruct_dataset, orcabest_dataset, expertqa_dataset], tokenizer)
        eval_dataset = None
    else:
        raise NotImplementedError
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
