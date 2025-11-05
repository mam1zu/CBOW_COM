import subprocess
import os
import sys
import numpy as np

CORPUS_SPLIT_SIZE = 2**30
CHUNKS_NUM = 30
LINES_PER_CHUNK=50000
#corpus_path = "/tf/paper/wikidata/wiki-cleaned.txt"
corpus_path = "/tf/paper/cbow-com/utils/wiki-cleaned.nostopword.100000.txt"
split_corpus_dir = "/tf/paper/cbow-com/dataset/wikidata/"
cmd = f"wc -l {corpus_path}" #任意コード実行の脆弱性になりそうで草
print("counting corpus lines")
line_count = subprocess.check_output(cmd.split()).decode().split()[0]

file_counter = 0
line_counter = 0
counter = 0
split_corpus_path = os.path.join(split_corpus_dir, f"wiki-cleaned.batch_{file_counter:03d}.txt")
file = open(split_corpus_path, 'w')
lines_per_chunk = 50000

with open(corpus_path, 'r') as f_corpus:
    for line in f_corpus:
        file.write(line)
        
        if line_counter >= lines_per_chunk:
            file.close()
            file_counter += 1
            split_corpus_path = os.path.join(split_corpus_dir, f"wiki-cleaned.batch_{file_counter:03d}.txt")
            file = open(split_corpus_path, 'w')

        counter += 1
        line_counter = line_counter + 1 if line_counter < lines_per_chunk else 0
        if counter % 100000 == 0:
            print(f"Processed {counter} lines")
    
    file.close()
