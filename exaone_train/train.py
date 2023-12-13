##    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Editor : Hyunjik Jo, Sunkyoung Kim, Eunbi Choi

from dataclasses import dataclass, field
from typing import Optional, Dict
import os

import transformers
from transformers import Trainer
from dataset import make_supervised_data_module

#from exaone_peft.model_utils import get_models

import os
os.environ['CURL_CA_BUNDLE'] = ''
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Directory or file path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Directory or file path to the evaluation data."}
    )
    data_class: str = field(
        default=None, metadata={"help": "Class name to the dataset."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    wandb_project_name: str = field(
        default=None, metadata={"help": "Name for the Wandb project"}
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    os.makedirs(output_dir, exist_ok=True)
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#    model, tokenizer = get_models('/workspace/home/code/exaone_peft/scripts/configs/finetune/bf16_full_finetuning.yaml')

#    wandb.init()
#    training_args.set_training(**wandb.config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
        truncation_side='left', # Eunbi
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = "<unk>"
    if not tokenizer.unk_token:
        tokenizer.unk_token = "<unk>"
    if not tokenizer.bos_token:
        tokenizer.bos_token = "<s>"
    if not tokenizer.eos_token:
        tokenizer.eos_token = "</s>"

#    os.environ["WANDB_PROJECT"] = training_args.wandb_project_name  # wandb 프로젝트 명
#    os.environ["WANDB_DISABLED"] = "true"                           # wandb on/off
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True
#    model.is_parallelizable = False # For GPT-2
#    model.model_parallel = False    # For GPT-2

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
