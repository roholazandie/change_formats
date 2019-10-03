import ast
import json
from collections import defaultdict
import random
import numpy as np
import time


def convert_to_json(filename):
    with open(filename) as file_reader:
        all_dialogs = []
        for line in file_reader:
            conversations = ast.literal_eval(line)
            all_dialogs.append(conversations)

    def random_select_conversation(all_dialogs):
        random_selected = np.random.choice(all_dialogs, 19)
        random_selected = [np.random.choice(dialog["dialogue"]) for dialog in random_selected]

        random_selected_texts = [item['text'] for item in random_selected]
        random_selected_emotions = ['<no_emotion>' if item['emotion'] == "no_emotion" else '<emotion>' for item in random_selected]
        random_selected_acts = ['<'+item['act']+'>' for item in random_selected]
        return random_selected_texts, random_selected_emotions, random_selected_acts

    dataset = defaultdict(list)
    dataset["train"] = []
    all_emotions = set([])
    all_acts = set([])
    for conversation in all_dialogs:
        conversation = conversation["dialogue"]
        utterances = []
        for i in range(len(conversation)-1):
            utterance = defaultdict(list)
            utterance["history"] = [conv['text'] for conv in conversation[:i+1]]
            utterance["emotion"] = ['<emotion>' if conv['emotion'] != "no_emotion" else '<no_emotion>' for conv in conversation[:i+1]]
            all_emotions = all_emotions.union(set(utterance["emotion"]))
            utterance["act"] = ['<'+conv['act']+'>' for conv in conversation[:i+1]]
            #all_acts = all_acts.union(set(utterance["act"]))
            true_conversation = conversation[i + 1]['text']
            emotion = conversation[i + 1]['emotion']
            true_emotion = '<no_emotion>' if emotion == "no_emotion" else '<emotion>'
            true_act = '<' + conversation[i + 1]['act'] + '>'
            random_selected_texts, random_selected_emotions, random_selected_acts = random_select_conversation(all_dialogs)
            utterance["candidates"] = random_selected_texts + [true_conversation]
            utterance["candidates_emotions"] = random_selected_emotions + [true_emotion]
            utterance["candidates_acts"] = random_selected_acts + [true_act]

            utterances.append(utterance)

        if utterances:
            dataset["train"].append({"personality": [], "utterances": utterances})

    print(all_emotions)
    print(all_acts)

    return dataset


def merge_dicts():
    with open("/home/rohola/data/daily_dialog_emotion_only/test_out.json") as file_reader:
        test_dict = json.load(file_reader)

    test_dict["valid"] = test_dict["train"]
    del test_dict["train"]
    with open("/home/rohola/data/daily_dialog_emotion_only/valid_out.json") as file_reader:
        valid_dict = json.load(file_reader)

    test_dict["valid"].extend(valid_dict["train"])

    with open("/home/rohola/data/daily_dialog_emotion_only/train_out.json") as file_reader:
        train_dict = json.load(file_reader)

    dataset_dict = {**train_dict, **test_dict}

    with open("/home/rohola/data/daily_dialog_emotion_only/daily_dialog.json", 'w') as file_writer:
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
    # filename = "/media/rohola/data/dialog_systems/daily_dialog/valid.json"
    # dataset = convert_to_json(filename)
    #
    # with open('/home/rohola/data/daily_dialog_emotion_only/valid_out.json', 'w') as fp:
    #     json.dump(dataset, fp)

    merge_dicts()
    #find_longest()