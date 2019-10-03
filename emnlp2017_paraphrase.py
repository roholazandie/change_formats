from itertools import chain
import numpy as np
from random import shuffle

def convert_emnlp2017(input_file, output_file):
    lines = open(input_file).readlines()
    lines = [line.split('\t') for line in lines]
    all_sentences = list(chain(*lines))
    with open(output_file, 'w') as fw:
        for line in lines:
            random_sentence = np.random.choice(all_sentences, 1)[0]
            fw.write(line[0].rstrip() + '\t' + line[1].rstrip() + '\t' + random_sentence.rstrip() + '\t0\n')


def split_dataset(output_file, train_file, eval_file):
    lines = open(output_file).readlines()
    shuffle(lines)
    n_samples = len(lines)
    train_size = np.floor(0.90*n_samples)

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
    input_file = "/media/rohola/data/paraphrase_datasets/paraphrase_dataset_emnlp2017/2016_Oct_10--2017_Jan_08_paraphrase.txt"
    output_file = "/media/rohola/data/paraphrase_datasets/paraphrase_dataset_emnlp2017/emnlp2017_paraphrase.txt"
    #convert_emnlp2017(input_file, output_file)

    train_file = "/media/rohola/data/paraphrase_datasets/paraphrase_dataset_emnlp2017/emnlp2017_train.txt"
    eval_file = "/media/rohola/data/paraphrase_datasets/paraphrase_dataset_emnlp2017/emnlp2017_eval.txt"
    split_dataset(output_file, train_file, eval_file)