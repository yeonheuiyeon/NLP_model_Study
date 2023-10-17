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
defalt_p="[INST] Is phrase 0 or phrase 1 appropriate to complete the given sentence? [/INST]\n\nSentences that need to be completed : To get cash for old, unused drinking bottles,\nCandidate phrase 0 : take the bottles to a grocery store with a recycling center.\nCandidate phrase 1 : stand on the street selling the bottles for a small price.\nAnswer : 0\n\n[INST] Is phrase 0 or phrase 1 appropriate to complete the given sentence? [/INST]\n\nSentences that need to be completed : how do you get out in dodgeball?\nCandidate phrase 0 : catch a ball.\nCandidate phrase 1 : have a ball hit you.\nAnswer : 1\n\n[INST] Is phrase 0 or phrase 1 appropriate to complete the given sentence? [/INST]\n\nSentences that need to be completed : To close a small, but deep wound off,\nCandidate phrase 0 : cauterize the wound with a heated piece of metal.\nCandidate phrase 1 : pour some salt into the wound to help it dry out.\nAnswer : 0\n\n[INST] Is phrase 0 or phrase 1 appropriate to complete the given sentence? [/INST]\n\nSentences that need to be completed : To trim away excess fondant used in Flower Pot Cupcakes.\nCandidate phrase 0 : Use small, sharp scissors to trim.\nCandidate phrase 1 : Use a small, sharp knife to trim.\nAnswer : 1"
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
    max_new_tokens=50,
)

bathchs_input=[]
labels=[]

dataset = datasets.load_dataset("commonsense_qa", split="train")
with open("commonsense_qa.jsonl", "w", encoding="utf-8") as f:
    for idx, data in enumerate(dataset):
        if len(bathchs_input)==4:
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
            for i in range(4):
                my_data = OrderedDict()
                my_data["instruction"] = bathchs_input[i]
                my_data["pred_llama"] = output[i]
                my_data["labels"] = labels[i]
                json.dump(my_data, f)
                f.write("\n")
            bathchs_input = []그럼 
            labels = []

        cotent = data['goal']
        sol1 = data['question']
        answer=data['label']
        prompt=f"{defalt_p}\n\n{B_INST} Find the correct answer to the question among the 5 given candidates {E_INST}\n\nQuestion : {sol1}\nCandidates : A) {} B) fatigue C) mercy D) empathy E) anxiety\nAnswer : "
        bathchs_input.append(prompt)
        labels.append(answer)










