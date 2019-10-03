import numpy as np
from random import shuffle
import language_check

def create_random_sentence_dataset(dataset_file, output_file):
    with open(output_file, 'w') as writer:
        with open(dataset_file) as f:
            for line in f:
                info = line.split("|||")
                if info[0].rstrip() == "[S]":
                    writer.write(info[1].rstrip()+"\n")
                    writer.write(info[2].rstrip()+"\n")


def create_ppdb_dataset(dataset_file, all_sentences_file, output_file):
    tool = language_check.LanguageTool('en-US')
    all_sentences = open(all_sentences_file).readlines()
    with open(output_file, 'w') as file_writer:
        with open(dataset_file) as file_reader:
            for line in file_reader:
                info = line.split("|||")
                if info[0].rstrip() == "[S]" and info[5].strip().lower() == "equivalence":
                    sentence1 = info[1]
                    sentence2 = info[2]
                    random_sentence = np.random.choice(all_sentences, 1)[0]
                    if '-' in sentence1:
                        sentence1 = sentence1.replace("-", "")
                        sentence1 = sentence1.strip()

                    if '-' in sentence2:
                        sentence2 = sentence2.replace("-", "")
                        sentence2 = sentence2.strip()

                    if '-' in random_sentence:
                        random_sentence = random_sentence.replace("-", "")
                        random_sentence = random_sentence.strip()
                    sentence1 = tool.correct(sentence1)
                    sentence2 = tool.correct(sentence2)
                    random_sentence = tool.correct(random_sentence)
                    file_writer.write(sentence1.strip() + "\t" + sentence2.strip() + "\t" + random_sentence.strip() + "\t0\n")



def split_dataset(output_file, train_file, eval_file):
    lines = open(output_file).readlines()
    shuffle(lines)
    n_samples = len(lines)
    train_size = np.floor(0.8*n_samples)

    train_lines = []
    eval_lines = []
    for i, line in enumerate(lines):
        if i < train_size:
            train_lines.append(line)
        else:
            eval_lines.append(line)

    open(train_file, 'w').writelines(train_lines)
    open(eval_file, 'w').writelines(eval_lines)




if __name__ == "__main__":
    ppdb_xxxl = "/media/rohola/data/PPDB/ppdb-2.0-xxxl-phrasal"
    ppdb_phrasal = "/media/rohola/data/PPDB/ppdb-2.0-m-phrasal"
    all_sentences_file = "/home/rohola/codes/change_formats/data/random_sentences.txt"
    output_file = "/home/rohola/codes/change_formats/data/ppdb_out.txt"
    train_file = "/home/rohola/codes/change_formats/data/ppdb_train.txt"
    eval_file = "/home/rohola/codes/change_formats/data/ppdb_eval.txt"
    #create_ppdb_dataset(ppdb_phrasal, all_sentences_file, output_file)
    #create_random_sentence_dataset(ppdb_xxxl, random_sentences_file)
    split_dataset(output_file, train_file, eval_file)
