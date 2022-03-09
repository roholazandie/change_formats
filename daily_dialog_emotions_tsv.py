import ast

def convert_to_tsv(filename, set_type):
    with open(filename) as file_reader:
        all_dialogs = []
        for line in file_reader:
            conversations = ast.literal_eval(line)
            all_dialogs.append(conversations)

    with open("/media/rohola/data/dialog_systems/daily_dialog/daily_dialog_emotions/"+set_type+".tsv", 'w') as file_writer:
        for dialog in all_dialogs:
            for conversation in dialog["dialogue"]:
                emotion = conversation["emotion"]
                text = conversation["text"]
                file_writer.write(text+'\t'+emotion+'\n')












if __name__ == "__main__":
    filename = "/media/rohola/data/dialog_systems/daily_dialog/test.json"
    dataset = convert_to_tsv(filename, set_type='test')