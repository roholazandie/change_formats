import numpy as np
from random import shuffle


def convert_msrp_format(msrp_file, group_file, out_file):
    group = {}
    with open(group_file) as fr:
        next(fr)
        for line in fr:
            value = int(line.split(',')[0].rstrip())
            key = int(line.split(',')[1].rstrip())
            if key in group:
                group[key].append(value)
            else:
                group[key] = [value]

    lines = open(msrp_file).readlines()
    lines = [line.rstrip() for line in lines]
    with open(out_file, 'w') as fw:
        for key in group:
            line_numbers = group[key]
            if len(line_numbers) == 3:
                random_sentence = np.random.choice(lines, 1)[0]
                fw.write(lines[line_numbers[0]].rstrip()+"\t"+lines[line_numbers[1]]+"\t"+random_sentence+ "\t0\n")
                random_sentence = np.random.choice(lines, 1)[0]
                fw.write(lines[line_numbers[0]].rstrip()+"\t"+lines[line_numbers[2]]+"\t"+random_sentence+ "\t0\n")
            elif len(line_numbers) == 4:
                random_sentence = np.random.choice(lines, 1)[0]
                fw.write(lines[line_numbers[0]].rstrip() + "\t" + lines[line_numbers[1]] + "\t" + random_sentence + "\t0\n")
                random_sentence = np.random.choice(lines, 1)[0]
                fw.write(lines[line_numbers[0]].rstrip() + "\t" + lines[line_numbers[2]] + "\t" + random_sentence + "\t0\n")
                random_sentence = np.random.choice(lines, 1)[0]
                fw.write(lines[line_numbers[0]].rstrip() + "\t" + lines[line_numbers[3]] + "\t" + random_sentence + "\t0\n")
            else:
                random_sentence = np.random.choice(lines, 1)[0]
                fw.write(lines[line_numbers[0]].rstrip() + "\t" + lines[line_numbers[1]] + "\t" + random_sentence + "\t0\n")
                random_sentence = np.random.choice(lines, 1)[0]
                fw.write(lines[line_numbers[0]].rstrip() + "\t" + lines[line_numbers[2]] + "\t" + random_sentence + "\t0\n")
                random_sentence = np.random.choice(lines, 1)[0]
                fw.write(lines[line_numbers[0]].rstrip() + "\t" + lines[line_numbers[3]] + "\t" + random_sentence + "\t0\n")
                random_sentence = np.random.choice(lines, 1)[0]
                fw.write(lines[line_numbers[0]].rstrip() + "\t" + lines[line_numbers[4]] + "\t" + random_sentence + "\t0\n")


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
    msrp_file = "/media/rohola/data/paraphrase_datasets/msrp_paraphrase_grouped/msrp_distribute/phrases.txt"
    group_file = "/media/rohola/data/paraphrase_datasets/msrp_paraphrase_grouped/msrp_distribute/phrase_groups.csv"
    out_file = "/media/rohola/data/paraphrase_datasets/msrp_paraphrase_grouped/msrp_distribute/out_file.txt"
    train_file = "/media/rohola/data/paraphrase_datasets/msrp_paraphrase_grouped/msrp_distribute/msrp_train.txt"
    eval_file = "/media/rohola/data/paraphrase_datasets/msrp_paraphrase_grouped/msrp_distribute/msrp_eval.txt"


    #convert_msrp_format(msrp_file, group_file, out_file)

    split_dataset(out_file, train_file, eval_file)
