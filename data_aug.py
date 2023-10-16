from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import copy
import datasets
import itertools
import torch
from transformers import LlamaTokenizer
from collections import OrderedDict
if torch.cuda.is_available():
    device = "cuda"
B_INST, E_INST = "[INST]", "[/INST]"
model = LlamaForCausalLM.from_pretrained("../llama-recipes/outpus",device_map="auto")
model.eval()
tokenizer = LlamaTokenizer.from_pretrained("../llama-recipes/outpus")
if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

tokenizer.padding_side = 'left'
generation_config = GenerationConfig(
    temperature=1,
    top_p=1,
    top_k=50,
    num_beams=1,
    max_new_tokens=128,
)

bathchs_input=[]
labels=[]

dataset = datasets.load_dataset("boolq", split="train")
with open("boolq_qna.jsonl", "w", encoding="utf-8") as f:
    for idx, data in enumerate(dataset):
        if len(bathchs_input)==8:
            inputs_b = tokenizer(bathchs_input, return_tensors="pt", padding=True).to(device)
            input_idsb = inputs_b["input_ids"].to(device)
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_idsb,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            s = generation_output.sequences
            output = tokenizer.batch_decode(s, skip_special_tokens=True)
            if idx<20:
                print(output)
            for i in range(8):
                my_data = OrderedDict()
                my_data["instruction"] = bathchs_input[i]
                my_data["pred_llama"] = output[i]
                my_data["labels"] = labels[i]
                json.dump(my_data, f)
                f.write("\n")
            bathchs_input = []
            labels = []

        cotent = data['passage']
        q = data['question']
        answer=data['answer']
        prompt=f"{B_INST} Read the passage and answer whether it is true or false. {E_INST}\n{cotent}\n\nQuestion:{q}\n\nThe answer is? :"
        bathchs_input.append(prompt)
        labels.append(answer)









