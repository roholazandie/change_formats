import json
import numpy as np
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from tqdm import tqdm
from itertools import chain
import torch
import spacy
import pickle


def random_select_conversation(all_dialogs, true_conversation):
    random_selected = np.random.choice(all_dialogs, 19)
    random_selected = [np.random.choice(list(dialog), 1)[0] for dialog in random_selected]
    return random_selected + [true_conversation]


def read_conversations(conversation_filename, reading_set_filename):
    with open(conversation_filename) as fr:
        conversations = json.load(fr)

    with open(reading_set_filename) as fr:
        knowledge = json.load(fr)

    a = 1
    all_lengths = [sum(
        [len(conversations[k]['content'][i]['message'].split(' ')) for i in range(len(conversations[k]['content']))])
                   for k in conversations]

    min_length_conversation = np.min(all_lengths)
    max_length_conversation = np.max(all_lengths)
    mean_length_conversaion = np.mean(all_lengths)
    print(min_length_conversation)
    print(max_length_conversation)
    print(mean_length_conversaion)

    num_bins = 100
    plt.hist(all_lengths, num_bins, facecolor='blue', alpha=0.5)
    plt.show()


def change_format(conversation_filename, reading_set_filename):
    with open(conversation_filename) as fr:
        conversations = json.load(fr)

    with open(reading_set_filename) as fr:
        knowledges = json.load(fr)


    all_conversations = []
    for key in conversations:
        conversation = conversations[key]

        conversation_list = []
        for content in conversation['content']:
            dialog = defaultdict(list)

            message = content["message"]
            sentiment = content["sentiment"]
            agent = content["agent"]
            dialog["message"] = message
            dialog["sentiment"] = sentiment
            dialog["agent"] = agent

            agent_knowledges = []
            agent_fun_facts = []
            topics = []
            for knowledge_id in content["knowledge_source"]:
                if knowledge_id in knowledges[key][agent]:
                    agent_knowledge = knowledges[key][agent][knowledge_id]
                    shortened_wiki_knowledge = ""
                    if "shortened_wiki_lead_section" in agent_knowledge:
                        shortened_wiki_knowledge = agent_knowledge["shortened_wiki_lead_section"]
                    if "summarized_wiki_lead_section" in agent_knowledge:
                        shortened_wiki_knowledge = agent_knowledge["summarized_wiki_lead_section"]

                    fun_facts = agent_knowledge["fun_facts"]
                    agent_knowledges.append(shortened_wiki_knowledge)
                    agent_fun_facts.extend(fun_facts)
                    topics.append(agent_knowledge["entity"])

            dialog["knowledges"] = agent_knowledges
            dialog["fun_facts"] = agent_fun_facts
            dialog["topics"] = list(set(topics))
            conversation_list.append(dialog)

        all_conversations.append(conversation_list)

    return all_conversations


if __name__ == "__main__":
    conversation_filename = "/media/rohola/data/dialog_systems/alexa_prize_topical_chat_dataset/conversations/train.json"
    reading_set_filename = "/media/rohola/data/dialog_systems/alexa_prize_topical_chat_dataset/reading_sets/train.json"
    dataset = change_format(conversation_filename, reading_set_filename)
    with open('/media/rohola/data/dialog_systems/alexa_prize_topical_chat_dataset/train.json', 'w') as fp:
        json.dump(dataset, fp)