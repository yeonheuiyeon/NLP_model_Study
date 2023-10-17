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
defalt_p=("[INST] Find the correct answer to the question among the 5 given candidates [/INST]\n\nQuestion : Too many people want exotic snakes. The demand is driving what to carry them?\nCandidates : A) ditch B) shop C) north america D) pet shops E) outdoors\nAnswer : D\n\n[INST] What are you hoping to do when listening to an expert speak? [/INST]\n\nQuestion : Too many people want exotic snakes. The demand is driving what to carry them?\nCandidates : A) learning B) fatigue C) mercy D) empathy E) anxiety\nAnswer : A\n\n[INST] Where can someone purchase a contraceptive device without a prescription? [/INST]\n\nQuestion : Too many people want exotic snakes. The demand is driving what to carry them?\nCandidates : A) pharmacy B) person C) drugstore D) bedroom E) mcdonalds\nAnswer : C\n\n[INST] Find the correct answer to the question among the 5 given candidates [/INST]\n\nQuestion : After the sun has risen what happens at the end of the day?\nCandidates : A) deep dive B) fall C) lower D) below E) sun set\nAnswer : E")
model = LlamaForCausalLM.from_pretrained("yeen214/test_llama2_7b",device_map="auto")
model.eval()
tokenizer = LlamaTokenizer.from_pretrained("yeen214/test_llama2_7b")
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

        cotent = data['choices']['text']
        sol1 = data['question']
        answer=data['answerKey']
        prompt=f"{defalt_p}\n\n{B_INST} Find the correct answer to the question among the 5 given candidates {E_INST}\n\nQuestion : {sol1}\nCandidates : A) {cotent[0]} B) {cotent[1]} C) {cotent[2]} D) {cotent[3]} E) {cotent[4]}\nAnswer : "
        bathchs_input.append(prompt)
        labels.append(answer)










