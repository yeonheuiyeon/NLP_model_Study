total=[]
import jsonlines
import re
import os
with jsonlines.open("v2.jsonl") as f:
    for line in f.iter():
        n_t=re.sub("\.txt|\.json|\.hwp|\.pdf","",line['title'])
        text=text.strip().split()
        split_text=len(text)
        if split_text<10:
            continue
        if split_text>300:
            tt=300
            while True:
                if split_text<=tt:
                    break
                tmp=text[:tt]
                new_t = f"\n = {n_t} = \n\n{' '.join(tmp)}\n"
                total.append(new_t)
                text=[tt:]
                split_text-=tt


        new_t=f"\n = {n_t} = \n\n{' '.join(text)}\n"
        total.append(new_t)



import random
random.shuffle(total)
train_text=''.join(total[:-20])
valid_text=''.join(total[-20:-10])
test_text=''.join(total[-10:])

with open("train_text.txt", "w") as f:
    f.write(train_text)
with open("valid_text.txt", "w") as f:
    f.write(valid_text)
with open("test_text.txt", "w") as f:
    f.write(test_text)
