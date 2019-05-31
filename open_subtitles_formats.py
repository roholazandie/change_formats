import json
from collections import defaultdict
import random
import numpy as np
import time
from random_sampler import select

with open("/media/rohola/data/dialog_systems/persona_chat/personachat_self_original.json") as f:
    result = json.load(f)

print(result["train"][0]["personality"])
print(result["train"][0]["utterances"][3]["candidates"])
print(result["train"][0]["utterances"][3]["history"])



def convert_to_json(filename):
    with open(filename) as file_reader:
        all_dialogs = []
        dialog = []
        for line in file_reader:
            number, rest = line.split(" ", 1)
            if int(number) == 1:
                all_dialogs.append(dialog)
                dialog = []

            parts = rest.split('\t')
            if len(parts) == 2:
                first, second = parts
                dialog_tuple = [first.rstrip(), second.rstrip()]
                dialog.extend(dialog_tuple)


        def random_select_conversation(true_conversation):
            random_selected = select(filename, k=19)
            return random_selected + [true_conversation]



        dataset = defaultdict(list)
        dataset["train"] = []

        #k = 0
        for dialog in all_dialogs:
            utterances = []
            #t1 = time.time()
            for i in range(0, len(dialog), 2):
                utterance = defaultdict(list)
                utterance["history"] = dialog[:i+1]
                true_conversation = dialog[i+1]
                utterance["candidates"] = random_select_conversation(true_conversation)
                utterances.append(utterance)

            if utterances:
                dataset["train"].append({"personality":[], "utterances": utterances})


            # k+=1
            # if k>1000:
            #     break
            #t2 = time.time()


        return dataset


if __name__ == "__main__":
    t1 = time.time()
    #filename = "/media/rohola/data/dialog_systems/opensubtitle/train.txt"
    filename = "/media/data/rohola_data/opensubtitles/test.txt"

    dataset = convert_to_json(filename)
    with open('/media/data/rohola_data/opensubtitles/test_out.json', 'w') as fp:
        json.dump(dataset, fp)
    t2 = time.time()
    print(t2-t1)