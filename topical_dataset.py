import json
import numpy as np
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import defaultdict
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
    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    nlp = spacy.load("en_pytt_bertbaseuncased_lg")

    with open(conversation_filename) as fr:
        conversations = json.load(fr)

    with open(reading_set_filename) as fr:
        knowledges = json.load(fr)

    all_dialogs = []
    all_dialog_knowledge = []
    all_dialog_fun_facts = []

    agent_knowledge_lengths = []

    for key in conversations:
        conversation = conversations[key]

        dialogs = []
        dialog_knowledge = []
        dialog_fun_facts = []

        for content in conversation['content']:

            message = content["message"]
            dialogs.append(message)
            sentiment = content["sentiment"]
            agent = content["agent"]

            agent_knowledge_string = ""
            agent_fun_facts = []
            for knowledge_id in content["knowledge_source"]:
                if knowledge_id in knowledges[key][agent]:
                    agent_knowledge = knowledges[key][agent][knowledge_id]
                    shortened_wiki_knowledge = ""
                    if "shortened_wiki_lead_section" in agent_knowledge:
                        shortened_wiki_knowledge = agent_knowledge["shortened_wiki_lead_section"]
                    if "summarized_wiki_lead_section" in agent_knowledge:
                        shortened_wiki_knowledge = agent_knowledge["summarized_wiki_lead_section"]

                    fun_facts = agent_knowledge["fun_facts"]
                    agent_knowledge_string += shortened_wiki_knowledge
                    agent_fun_facts.extend(fun_facts)

                # if knowledge_id in knowledges[key]["article"]:
                #     article = knowledges[key]["article"][knowledge_id]
                #     agent_knowledge_string += article

            dialog_knowledge.append(agent_knowledge_string)
            dialog_fun_facts.append(agent_fun_facts)
            agent_knowledge_lengths.append(len(agent_knowledge_string.split(' ')))

        assert len(dialog_fun_facts) == len(dialogs)
        assert len(dialog_knowledge) == len(dialogs)
        all_dialogs.append(dialogs)
        all_dialog_knowledge.append(dialog_knowledge)
        all_dialog_fun_facts.append(dialog_fun_facts)

    all_dialog_embeds = []
    for dialogs in tqdm(all_dialogs):
        dialog_embed = []
        for message in dialogs:
            doc = nlp(message)
            dialog_embed.append(doc.vector)
        all_dialog_embeds.append(dialog_embed)

    all_fun_facts_embed = []
    for fun_facts in tqdm(all_dialog_fun_facts):
        fun_facts_embed = []
        for facts in fun_facts:
            if facts:
                fun_facts_embed.append([nlp(fact).vector for fact in facts])
            else:
                fun_facts_embed.append([])

        all_fun_facts_embed.append(fun_facts_embed)

    all_dialog_knowledge_sents = []
    all_dialog_knowledge_embed = []
    for dialog_knowledge in tqdm(all_dialog_knowledge):
        dialog_knowledge_sent = []
        dialog_knowledge_embed = []
        for knowledge in dialog_knowledge:
            if knowledge.rstrip():
                doc = nlp(knowledge)
                sentences_embeds = [sent.vector for sent in doc.sents]
                dialog_knowledge_embed.append(sentences_embeds)
                dialog_knowledge_sent.append([sent.text for sent in doc.sents])
            else:
                dialog_knowledge_embed.append([])
                dialog_knowledge_sent.append([])
        all_dialog_knowledge_sents.append(dialog_knowledge_sent)
        all_dialog_knowledge_embed.append(dialog_knowledge_embed)

    # num_bins = 100
    # plt.hist(agent_knowledge_lengths, num_bins, facecolor='blue', alpha=0.5)
    # plt.show()

    dataset = defaultdict(list)
    dataset["train"] = []

    for dialogs, dialog_embeds, fun_facts, fun_facts_embed, dialog_knowledge_sents, dialog_knowledge_embed in zip(
            all_dialogs, all_dialog_embeds, all_dialog_fun_facts, all_fun_facts_embed, all_dialog_knowledge_sents,
            all_dialog_knowledge_embed):
        utterances = []
        for i in range(len(dialogs) - 1):
            utterance = defaultdict(list)
            utterance["history"] = dialogs[:i + 1]
            true_conversation = dialogs[i + 1]
            utterance["candidates"] = random_select_conversation(all_dialogs, true_conversation)

            utterance["fun_fact"] = fun_facts[i]
            utterance["fun_fact_embed"] = fun_facts_embed[i]  # [arr.tolist() for arr in ]
            utterance["knowledge"] = dialog_knowledge_sents[i]
            utterance["knowledge_embed"] = dialog_knowledge_embed[
                i]  # [arr.tolist() for arr in dialog_knowledge_embed[i]]

            utterances.append(utterance)

        dataset["train"].append(utterances)

    return dataset


if __name__ == "__main__":
    conversation_filename = "/media/rohola/data/dialog_systems/alexa_prize_topical_chat_dataset/conversations/train.json"
    reading_set_filename = "/media/rohola/data/dialog_systems/alexa_prize_topical_chat_dataset/reading_sets/train.json"
    # # read_conversations(conversation_filename, reading_set_filename)
    dataset = change_format(conversation_filename, reading_set_filename)
    # pickle.dump(dataset, open("dataset.p", "wb"))

    # train_dict = pickle.load(
    #     open("/media/data2/rohola_data/datasets/alexa_prize_topical_chat_dataset/out/train_dataset_bert.p", 'rb'))
    # valid_dict = pickle.load(
    #     open("/media/data2/rohola_data/datasets/alexa_prize_topical_chat_dataset/out/valid_dataset_bert.p", 'rb'))
    # dataset_dict = {**train_dict, **valid_dict}
    # pickle.dump(dataset_dict,
    #             open("/media/data2/rohola_data/datasets/alexa_prize_topical_chat_dataset/out/dataset_bert", "wb"))

    # dataset = pickle.load(open("/media/data2/rohola_data/datasets/alexa_prize_topical_chat_dataset/out/dataset_bert", "wb"))
    #
    # small_train_data = dataset['train'][:30]
    # small_valid_data = dataset['valid'][:10]
    # small_dataset_dict = {"train": small_train_data, "valid": small_valid_data}
    # pickle.dump(small_dataset_dict, open("/media/data2/rohola_data/datasets/alexa_prize_topical_chat_dataset/out/small_dataset_bert", "wb"))