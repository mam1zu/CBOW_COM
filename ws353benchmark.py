#input: wordsim353 crowd bencnmark file
#output: rank of ws353c human value(pandas.DataFrame)
import pandas as pd
import csv

def ws353_dataframe() :

    data = {}
    df = pd.DataFrame({'word1': [], 'word2': [], 'human_value': []}, index=[])
    counter = 0
    try:
        with open('./benchmark/wordsim353.csv', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for line in reader:
                df.loc[counter] = line
                counter += 1

    except FileNotFoundError as e:
        print(f"{e}")
        return None

    #print(df.rank(numeric_only=True))
    return df

def simlex999_dataframe():
    data = {}
    df = pd.DataFrame({'word1': [], 'word2': [], 'human_value': []}, index=[])
    counter = 0
    try:
        with open('./benchmark/simlex999.csv', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for line in reader:
                df.loc[counter] = line
                counter += 1
    except FileNotFoundError as e:
        print(f"{e}")
        return None
    
    return df


def get_dataframe(benchmark_type: str):
    benchmarks = {'simlex999': 'simlex999.csv', 'ws353': 'wordsim353.csv', 'men': 'MEN.csv', 'ws353c': 'wordsim353crowd.csv', 'ws353r': 'wordsim353r.csv', 'ws353s': 'wordsim353s.csv'}
    if benchmark_type not in benchmarks:
        return None
    counter = 0
    df = pd.DataFrame({'word1': [], 'word2': [], 'human_value': []})
    try:
        with open(f'./benchmark/{benchmarks[benchmark_type]}', 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                df.loc[counter] = line
                counter += 1
    except FileNotFoundError as e:
        print(f"{e}")
        return None
    return df



def check_vocab(vocab_size) :
    with open(f'./utils/wiki-cleaned.nostopword.{vocab_size}.vocab', 'r') as vocab_file, open("./benchmark/simlex999.csv", 'r') as f:
        vocab_list = []
        nonvocab_list = []
        for line in vocab_file:
            word = line.split()
            vocab_list.append(word[0])
        reader = csv.reader(f)
        print("These words are not in the dictonary:")
        for line in reader:
            if line[0] not in vocab_list and line[0] not in nonvocab_list:
                nonvocab_list.append(line[0])
            if line[1] not in vocab_list and line[1] not in nonvocab_list:
                nonvocab_list.append(line[1])
        
        nonvocab_list.sort()
        print(nonvocab_list)

def check_vocab_men(vocab_size) :
    with open(f'./utils/wiki-cleaned.nostopword.{vocab_size}.vocab', 'r') as vocab_file, open("./benchmark/MEN.csv", 'r') as f:
        vocab_list = []
        nonvocab_list = []
        for line in vocab_file:
            word = line.split()
            vocab_list.append(word[0])
        reader = csv.reader(f)
        print("These words are not in the dictonary:")
        for line in reader:
            if line[0].lower() not in vocab_list and line[0] not in nonvocab_list:
                nonvocab_list.append(line[0])
            if line[1].lower() not in vocab_list and line[1] not in nonvocab_list:
                nonvocab_list.append(line[1])
        
        nonvocab_list.sort()
        print(nonvocab_list)
        #these are words not in vocab as of 2025/11/05
        #can, fire, mill, show, orthodontist, suds

if __name__ == '__main__':
    check_vocab_men(100000)