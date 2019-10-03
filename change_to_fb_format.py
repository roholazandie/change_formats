import json


with open("/home/rohola/data/daily_dialog.json") as fr:
    dataset = json.load(fr)

with open("daily_dialog_fb_format_train.txt", 'w') as file_writer:
    for item in dataset["train"]:
        for utterance in item["utterances"]:
            for i, dialog_turn in enumerate(utterance["history"], 1):
                if i == len(utterance["history"]):
                    label = utterance["candidates"][-1]
                    line = str(i) +" "+ dialog_turn + '\t' +label + '\t'+ "1" + '\t' + "|".join(utterance["candidates"]) +"\n"
                else:
                    line = str(i) + " "+ dialog_turn +'\n'

                file_writer.write(line)



with open("daily_dialog_fb_format_valid.txt", 'w') as file_writer:
    for item in dataset["valid"]:
        for utterance in item["utterances"]:
            for i, dialog_turn in enumerate(utterance["history"], 1):
                if i == len(utterance["history"]):
                    label = utterance["candidates"][-1]
                    line = str(i) +" "+ dialog_turn + '\t' +label + '\t'+ "1" + '\t' + "|".join(utterance["candidates"]) +"\n"
                else:
                    line = str(i) + " "+ dialog_turn +'\n'

                file_writer.write(line)