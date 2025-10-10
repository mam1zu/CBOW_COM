import subprocess
import os
import sys

CORPUS_SPLIT_SIZE = 2**30
corpus_path = "/tf/paper/cbow-com/wiki-cleaned.nostopword.100000.txt"
split_corpus_dir = "/tf/paper/cbow-com/dataset/wikidata/"
cmd = f"wc -l {corpus_path}" #任意コード実行の脆弱性で草
print("counting corpus lines")
line_count = subprocess.check_output(cmd.split()).decode().split()[0]

file_counter = 0
counter = 0
split_corpus_path = os.path.join(split_corpus_dir, f"wiki-cleaned.nostopword.100000.batch_{file_counter}.txt")
file = open(split_corpus_path, 'w')
with open(corpus_path, 'r') as f_corpus:
    for line in f_corpus:
        file.write(line)
        
        
        if os.path.getsize(split_corpus_path) > CORPUS_SPLIT_SIZE:
           file.close()
           file_counter += 1
           split_corpus_path = os.path.join(split_corpus_dir, f"wiki-cleaned.nostopword.100000.batch_{file_counter}.txt")
           file = open(split_corpus_path, 'w')
        counter += 1
        if counter % 100000 == 0:
            print(f"Processed {counter} lines")
    
    file.close()
