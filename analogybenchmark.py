from utils import load_dict, load_vocab
import csv
import glob
import sys
error_counter = 0
word_to_id, id_to_word = load_dict("./utils/wiki-cleaned.nostopword.100000.vocab")
vocab_list = load_vocab("./utils/wiki-cleaned.n
seed = ostopword.100000.vocab")
word_notfffffffin_vocab = []
files_path = glob.glob("./benchmark/google_analogy_dataset/*.csv")

#for file_path in files_path:
#    print(file_path)
#    with open(file_path, 'r') as file:
#        reader = csv.reader(file)
#        for line in reader:
#            for word in line:
#                if word not in vocab_list and word not in word_notin_vocab:
#                    word_notin_vocab.append(word)
#print(word_notin_vocab)

def get_file_list():
    return files_path

def get_benchmark_dataset():
    benchmark_datasets = []
    for enum, file_path in enumerate(files_path):
        print(f"{enum+1}. {file_path}")
        with open(file_path, 'r') as file:
            dataset = []
            reader = csv.reader(file)
            for line in reader:
                #print(f"{word_to_id[line[0]]} {word_to_id[line[1]]} {word_to_id[line[2]]} {word_to_id[line[3]]}")
                dataset.append([word_to_id[line[0]], word_to_id[line[1]], word_to_id[line[2]], word_to_id[line[3]] ])
            benchmark_datasets.append(dataset)

    return benchmark_datasets

if __name__ == "__main__":

    with open('./benchmark/google_analogy_dataset/gram1-adjective-to-adverb.csv') as file:
        reader = csv.reader(file)
        for line in reader:
            try:
                a = word_to_id[line[0]]
                b = word_to_id[line[1]]
                c = word_to_id[line[2]]
                d = word_to_id[line[3]]
            except KeyError:
                error_counter += 1
                continue
            print(f"{a} {b} {c} {d}")
        print(f"error counter: {error_counter}")


