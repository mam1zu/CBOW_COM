from scipy import sparse
import numpy as np
import torch
import sys
import os
import struct

COMAT_FILE_PATH = "./cooccurences.100000.bin"
DST_FILE_PATH = "./comat.100000"
RECORD_SIZE = 16
VOCAB_SIZE = 100000

rows = []
cols = []
data = []

try:
    comat_bin = open(COMAT_FILE_PATH, 'rb')
except FileNotFoundError as e:
    print(e)
    sys.exit(-1)

comat_lil = sparse.lil_matrix((100000, 100000), dtype=np.float32)
print(comat_lil)
counter = 0
while True:
    record = comat_bin.read(RECORD_SIZE)
    if not record:
        break
    word_1, word_2, value = struct.unpack('<iid', record)
    rows.append(word_1 - 1)
    cols.append(word_2 - 1)
    data.append(np.float32(value))
    #comat_lil[word_1 - 1, word_2 - 1] = cooccur_value
    counter += 1

    if counter % 10000000 == 0:
        print(f"processed {counter} tokens")
    
print(f"processed {counter} tokens")
print(f"Total {counter} un-zero records.")
comat_coo = sparse.coo_matrix((data, (rows, cols)), shape=(VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)


comat_csr = comat_coo.tocsr()

sparse.save_npz(DST_FILE_PATH, comat_csr)