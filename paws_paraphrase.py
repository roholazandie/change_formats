import numpy as np


def convert_paws_format(paws_file, out_file):
    paws_lines = open(paws_file).readlines()

    all_sentences = [p.split('\t')[1] for p in paws_lines]
    all_sentences.extend([p.split('\t')[2] for p in paws_lines])


    with open(out_file, 'w') as fw:
        for i, line in enumerate(paws_lines):
            if i == 0:
                continue
            line = line.split('\t')
            if int(line[-1].rstrip()) == 1:
                random_sentence = np.random.choice(all_sentences, 1)[0]
                fw.write(line[1].rstrip() + '\t' + line[2].rstrip() + '\t' + random_sentence + "\t0\n")


def merge(paws_train_file, paws_test_file):
    paws_train = open(paws_train_file).readlines()
    paws_test = open(paws_test_file).readlines()
    merged = paws_train + paws_test
    with open(paws_train_file, 'w') as fw:
        for line in merged:
            fw.write(line + "\n")


if __name__ == "__main__":
    paws_train_file = "/media/rohola/data/paraphrase_datasets/paws_wiki_labeled_final/final/train.tsv"
    paws_dev_file = "/media/rohola/data/paraphrase_datasets/paws_wiki_labeled_final/final/dev.tsv"
    paws_test_file = "/media/rohola/data/paraphrase_datasets/paws_wiki_labeled_final/final/test.tsv"

    out_train_file = "/media/rohola/data/paraphrase_datasets/paws_wiki_labeled_final/final/paws_train.txt"
    out_eval_file = "/media/rohola/data/paraphrase_datasets/paws_wiki_labeled_final/final/paws_eval.txt"
    #convert_paws_format(paws_dev_file, out_eval_file)

    merge(paws_train_file, paws_test_file)