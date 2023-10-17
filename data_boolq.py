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
defalt_p="[INST] Is the given question consistent (true) or inconsistent (false) with the given article? [/INST]\n\nPowdered sugar, also called confectioners' sugar, icing sugar, and icing cake, is a finely ground sugar produced by milling granulated sugar into a powdered state. It usually contains a small amount of anti-caking agent to prevent clumping and improve flow. Although most often produced in a factory, powdered sugar can also be made by processing ordinary granulated sugar in a coffee grinder, or by crushing it by hand in a mortar and pestle.\n\nQuestion : is confectionary sugar the same as powdered sugar\nTrue or False? : true\n\n[INST] Is the given question consistent (true) or inconsistent (false) with the given article? [/INST]\n\nBecause Earth and the Sun exhibits less gravitational pull than that of Krypton, and also due to his solar-powered body, the Man of Steel can also alter his personal mono-directional gravity field to propel himself through the air at will. Originally, he only had the power to jump great distances, as stated by the 1940s Superman cartoons slogan ``Able to leap tall buildings in a single bound.'' This was also shown in the movie Man of Steel. His power of flight has ranged from simply being able to jump great distances using his vast strength, to beginning in late 1941 being able to accelerate, float in midair, and change direction while traveling. Later he became able to traverse interstellar distances without stopping.\n\nQuestion : is superman's flight a feat of strength\nTrue or False? : false\n\n[INST] Is the given question consistent (true) or inconsistent (false) with the given article? [/INST]\n\nThe South Shetland Islands are a group of Antarctic islands, lying about 120 kilometres (75 mi) north of the Antarctic Peninsula, with a total area of 3,687 square kilometres (1,424 sq mi). By the Antarctic Treaty of 1959, the islands' sovereignty is neither recognized nor disputed by the signatories and they are free for use by any signatory for non-military purposes.\n\nQuestion : are the south shetland islands part of antarctica\nTrue or False? : true"
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

dataset = datasets.load_dataset("boolq", split="train")
with open("boolq_qna.jsonl", "w", encoding="utf-8") as f:
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
            bathchs_input = []
            labels = []

        cotent = data['passage']
        q = data['question']
        answer=data['answer']
        prompt=f"{defalt_p}\n\n{B_INST} Is the given question consistent (true) or inconsistent (false) with the given article? {E_INST}\n\n{cotent}\n\nQuestion : {q}\n\nTrue or False? : "
        bathchs_input.append(prompt)
        labels.append(answer)










