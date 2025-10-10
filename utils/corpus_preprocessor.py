import sys
import os
import operator
import numpy as np
from _stop_words import ENGLISH_STOP_WORDS

VOCAB_SIZE = 400000
SRC_FILE_PATH = f"../../wikidata/wiki-cleaned.txt" #wiki-cleaned
VCB_FILE_PATH = f"./wiki-cleaned.nostopword.400000.vocab" #wiki-cleaned.
VCBC_FILE_PATH = f"./wiki-cleaned.nostopword.400000.vocabc"
DST_FILE_PATH = f"./wiki-cleaned.nostopword.{VOCAB_SIZE}.txt" #wiki-cleaned.nostopword.[VOCAB_SIZE].txt

vocab_dict = {}
vocab_array = []

src_file = None
vcb_file = None
vcbc_file = None
dst_file = None

vcb_file_exists = False
vcbc_file_exists = False
try:
    src_file = open(SRC_FILE_PATH, 'r', encoding='utf-8')

    if not os.path.isfile(VCB_FILE_PATH):
        vcb_file = open(VCB_FILE_PATH, 'w', encoding='utf-8')
    else:
        vcb_file = open(VCB_FILE_PATH, 'r', encoding='utf-8')
        vcb_file_exists = True
    
    if not os.path.isfile(VCBC_FILE_PATH):
        vcbc_file = open(VCBC_FILE_PATH, 'w', encoding='utf-8')
    else:
        vcbc_file = open(VCBC_FILE_PATH, 'r', encoding='utf-8')
        vcbc_file_exists = True
    
    dst_file = open(DST_FILE_PATH, 'w', encoding='utf-8')

except FileExistsError as e:
    print(f"File already exists. {e}")
    sys.exit(-1)

except FileNotFoundError as e:
    print(f"Source File not found. {e}")
    sys.exit(-2)

def file_close() -> None:
    if src_file is not None:
        src_file.close()
    if vcb_file is not None:
        vcb_file.close()
    if vcbc_file is not None:
        vcbc_file.close()
    if dst_file is not None:
        dst_file.close()

#Generator
# Returns None if EOF
# Returns ""(empty str) if newline
# Returns word otherwise 
def get_word():
    while True:
        line = src_file.readline()
        if not line:
            yield None
            break
        for word in line.split():
            yield word
        yield ""

def vocab_count() -> None:
    counter = 0
    ch_generator = get_word()
    while True:
        try:
            ch = next(ch_generator)
        except StopIteration:
            break
        if ENGLISH_STOP_WORDS.__contains__(ch):
            #STOP WORD DETECTED, NOT COUNT
            continue
        if ch == "":
            #New line detected, continue
            continue
        if ch is None:
            #EOF detected, break
            break

        if ch == "<UNK>":
            print("Error, <UNK> vectors found in corpus.")
        
        if vocab_dict.get(ch) is None:
            vocab_dict[ch] = 1
        else:
            vocab_dict[ch] += 1
        
        counter += 1
        if counter % 10000000 == 0:
            print(f"\rprocessed {counter} tokens")
        
    print(f"processed {counter} tokens")
    
    #dict -> array
    for word, count in vocab_dict.items():
        vocab_array.append([word, count])

    vocab_array.sort(key=lambda x: (-x[1], x[0]))
    return vocab_array

def write_vocab_files(vocab_array) -> None:

    for word, count in vocab_array:
        vcbc_file.write(f"{word} {count}\n")
        vcb_file.write(f"{word}\n")

def load_vocabc_file() -> list :
    vocab_array = []
    while True:
        line = vcbc_file.readline()
        if not line:
            break
        count_data = line.split()
        vocab_array.append(count_data)

    return vocab_array

def write_new_corpus(vocab_dict) -> None:
    print(vocab_dict)
    counter = 0
    src_file.seek(0)
    ch_generator = get_word()
    while True:
        try:
            ch = next(ch_generator)
        except StopIteration:
            break
        
        if ch == "":
            #New line detected, continue
            dst_file.write("\n")
            continue
        
        if ch is None:
            #EOF detected, break
            dst_file.write("\n")
            break

        #Check if a word is in the dict
        if vocab_dict.get(ch) is None:
            continue

        dst_file.write(f"{ch} ")
    
        counter += 1
        if counter % 10000000 == 0:
            print(f"\rprocessed {counter} tokens")
        
    print(f"processed {counter} tokens")


def main():
    print("Corpus Preprocessor")

    print("Phase 1: Counting vocabulary")
    if vcb_file_exists and vcbc_file_exists:
        print("Vocabulary file detected, loading...")
        vocab_array = load_vocabc_file()
    else:
        vocab_array = vocab_count() #array of [word, count]
        vocab_array = vocab_array[:VOCAB_SIZE] #Cut vocabulary
        write_vocab_files(vocab_array)
        print("Phase 1: Counting finished and saved.")
    
    print("Phase 1: Done.")
    print("Phase 2: Writing new corpus using limited vocabulary data.")

    print(vocab_array)

    vocab_dict = {}
    
    for element in vocab_array:
        vocab_dict[element[0]] = element[1]
    
    write_new_corpus(vocab_dict)
    print("Phase 2: Done, processed corpus generated.")

    file_close()

if __name__ == "__main__":
    main()
