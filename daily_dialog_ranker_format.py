import json

with open("/home/rohola/data/daily_dialog_full/train_daily_dialog_ranker_format.txt", 'w') as file_writer:
    with open("/home/rohola/data/daily_dialog_full/train_out.json") as file_reader:
        daily_dialog_dict = json.load(file_reader)
        train_daily_dialog = daily_dialog_dict["train"]
        for diag in train_daily_dialog:
            for utterance in diag["utterances"]:
                conversation = utterance["history"]
                label = utterance["candidates"][-1]
                file_writer.write("text:"+ " ".join(conversation)+"\t" + "labels:"+label + "\t" +"episode_done:True \n")


with open("/home/rohola/data/daily_dialog_full/test_daily_dialog_ranker_format.txt", 'w') as file_writer:
    with open("/home/rohola/data/daily_dialog_full/test_out.json") as file_reader:
        daily_dialog_dict = json.load(file_reader)
        train_daily_dialog = daily_dialog_dict["train"]
        for diag in train_daily_dialog:
            for utterance in diag["utterances"]:
                conversation = utterance["history"]
                label = utterance["candidates"][-1]
                file_writer.write("text:"+ " ".join(conversation)+"\t" + "labels:"+label + "\t" +"episode_done:True \n")


with open("/home/rohola/data/daily_dialog_full/valid_daily_dialog_ranker_format.txt", 'w') as file_writer:
    with open("/home/rohola/data/daily_dialog_full/valid_out.json") as file_reader:
        daily_dialog_dict = json.load(file_reader)
        train_daily_dialog = daily_dialog_dict["train"]
        for diag in train_daily_dialog:
            for utterance in diag["utterances"]:
                conversation = utterance["history"]
                label = utterance["candidates"][-1]
                file_writer.write(
                    "text:" + " ".join(conversation) + "\t" + "labels:" + label + "\t" + "episode_done:True \n")