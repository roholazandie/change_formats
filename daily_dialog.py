import ast
import json
from collections import defaultdict
import random
import numpy as np
import time

def select(filename, k=3):
    samples = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()

        random_set = sorted(random.sample(range(file_size), k))

        i = 0
        cnt = k
        while cnt > 0:

            f.seek(random_set[i])
            f.readline()
            line = f.readline().strip()
            line = line.decode("utf-8").strip()

            if line:
                line_dict = ast.literal_eval(line)
                dialogue = line_dict["dialogue"]
                j = np.random.randint(len(dialogue))

                samples.append(dialogue[j]["text"])
                i += 1
                cnt -= 1
            else:
                print("--")

    assert len(samples) == k, "sample sizes are different"

    return samples



def convert_to_json(filename):
    with open(filename) as file_reader:
        all_dialogs = []
        for line in file_reader:
            conversations = ast.literal_eval(line)
            dialog = []
            if len(conversations["dialogue"]) % 2 == 1:
                conversations["dialogue"] = conversations["dialogue"][:-1]
            for i in range(0, len(conversations["dialogue"]), 2):
                first = conversations["dialogue"][i]["text"]
                second = conversations["dialogue"][i + 1]["text"]
                dialog_tuple = [first.rstrip(), second.rstrip()]
                dialog.extend(dialog_tuple)

            all_dialogs.append(dialog)

    def random_select_conversation(all_dialogs, true_conversation):
        #random_selected = select(filename)
        random_selected = np.random.choice(all_dialogs, 19)
        random_selected = [np.random.choice(list(dialog), 1)[0] for dialog in random_selected]
        return random_selected + [true_conversation]

    dataset = defaultdict(list)
    dataset["train"] = []

    # k = 0
    for dialog in all_dialogs:
        utterances = []
        t1 = time.time()
        for i in range(0, len(dialog), 2):
            utterance = defaultdict(list)
            utterance["history"] = dialog[:i + 1]
            true_conversation = dialog[i + 1]
            utterance["candidates"] = random_select_conversation(all_dialogs, true_conversation)
            utterances.append(utterance)

        if utterances:
            dataset["train"].append({"personality": [], "utterances": utterances})

        t2 = time.time()

        print(t2-t1)

    return dataset


def merge_dicts():


    with open("/home/rohola/data/test_out.json") as file_reader:
        test_dict = json.load(file_reader)

    test_dict["valid"] = test_dict["train"]
    del test_dict["train"]
    with open("/home/rohola/data/valid_out.json") as file_reader:
        valid_dict = json.load(file_reader)

    test_dict["valid"].extend(valid_dict["train"])

    with open("/home/rohola/data/train_out.json") as file_reader:
        train_dict = json.load(file_reader)

    dataset_dict = {**train_dict, **test_dict}

    with open("/home/rohola/data/daily_dialog.json", 'w') as file_writer:
        json.dump(dataset_dict, file_writer)


def find_longest():
    with open("/home/rohola/data/daily_dialog.json") as fr:
        dataset = json.load(fr)


    for item in dataset["train"]:
        for ut in item["utterances"]:
            for his in ut["history"]:
                if len(his.split()) > 300:
                    print(his)



if __name__ == "__main__":
    filename = "/media/rohola/data/dialog_systems/daily_dialog/test.json"
    dataset = convert_to_json(filename)
    #
    # with open('/media/data/rohola_data/daily_dialog/test_out.json', 'w') as fp:
    #     json.dump(dataset, fp)

    #merge_dicts()
    #find_longest()