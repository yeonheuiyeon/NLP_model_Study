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
defalt_p="[INST] Read the passage and answer the questions appropriately. [/INST]\n\nSince 1934, the sale of Federal Duck Stamps has generated $670 million, and helped to purchase or lease 5,200,000 acres (8,100 sq mi; 21,000 km2) of habitat. The stamps serve as a license to hunt migratory birds, an entrance pass for all National Wildlife Refuge areas, and are also considered collectors items often purchased for aesthetic reasons outside of the hunting and birding communities. Although non-hunters buy a significant number of Duck Stamps, eighty-seven percent of their sales are contributed by hunters, which is logical, as hunters are required to purchase them. Distribution of funds is managed by the Migratory Bird Conservation Commission (MBCC).\n\nQuestion : How many acres has the sale of Federal Duck Stamps helped to purchase or lease since 1934?\nAnswer : 5,200,000 acres\n\n[INST] Read the passage and answer the questions appropriately. [/INST]\n\nSerological methods are highly sensitive, specific and often extremely rapid tests used to identify microorganisms. These tests are based upon the ability of an antibody to bind specifically to an antigen. The antigen, usually a protein or carbohydrate made by an infectious agent, is bound by the antibody. This binding then sets off a chain of events that can be visibly obvious in various ways, dependent upon the test. For example, \"Strep throat\" is often diagnosed within minutes, and is based on the appearance of antigens made by the causative agent, S. pyogenes, that is retrieved from a patients throat with a cotton swab. Serological tests, if available, are usually the preferred route of identification, however the tests are costly to develop and the reagents used in the test often require refrigeration. Some serological methods are extremely costly, although when commonly used, such as with the \"strep test\", they can be inexpensive.\n\nQuestion : What is the causative agent of \"strep throat\"?\nAnswer : S. pyogenes\n\n[INST] Read the passage and answer the questions appropriately. [/INST]\n\nThe university owns several centers around the world used for international studies and research, conferences abroad, and alumni support. The university has had a presence in London, England, since 1968. Since 1998, its London center has been based in the former United University Club at 1 Suffolk Street in Trafalgar Square. The center enables the Colleges of Arts & Letters, Business Administration, Science, Engineering and the Law School to develop their own programs in London, as well as hosting conferences and symposia. Other Global Gateways are located in Beijing, Chicago, Dublin, Jerusalem and Rome.\n\nQuestion : At which location is the London Center operated by Notre Dame found?\nAnswer : 1 Suffolk Street in Trafalgar Square"
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

dataset = datasets.load_dataset("squad_v2", split="train")
with open("squadv2_qna.jsonl", "w", encoding="utf-8") as f:
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

        cotent = data['context']
        q = data['question']
        answer=data['answers']
        prompt=f"{defalt_p}\n\n{B_INST} Read the passage and answer the questions appropriately. {E_INST}\n\n{cotent}\n\nQuestion : {q}\nAnswer : "
        bathchs_input.append(prompt)
        labels.append(answer)










