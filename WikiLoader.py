import os
import sys
import random
import glob
import torch
import math
from utils import load_dict
from torch.utils.data import Dataset, DataLoader
import numpy as np

"""
WikiLoader.py
実装にあたっては, 
https://github.com/luffycodes/attention-word-embedding/blob/public_master/cbow.py
の内, 142行目〜348行目までの, CBOWDatasetクラスを参考にしたので, その旨記載する.
"""

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #data = {'centre': self.data[idx][0], 'left': self.data[idx][1: self.window_size], 'right': self.data[idx][self.window_size+1 : self.window_size * 2]}
        sample = {'data': self.data[idx], 'labels': self.labels[idx]}
        return sample

class CBOWDataset(Dataset):
    """
    基本的な動作まとめ

    ・gensimのWikiCorpusで処理した後のコーパスを処理対象とする
    ・記事毎に改行されているので, 1行が1記事を示している
    ・
    """

    def __init__(self, path, window_size, num_lines, num_line_per_chunk, word_vocab=None):
        self.window_size = window_size
        self.word_vocab = word_vocab
        self.num_lines = num_lines
        self.num_line_per_chunk = num_line_per_chunk #192433
        self.word_to_id, self.id_to_word = load_dict("/tf/paper/cbow-com/utils/wiki-cleaned.nostopword.100000.vocab")
        texts_generator = _generate_texts(path, num_line_per_chunk)


        if word_vocab is None:
            word_vocab = _generate_vocab()
        
        self.num_lines = 5772938
        self.num_chunks = math.ceil(self.num_lines / (1.0*self.num_line_per_chunk))
    
    def __getitem__(self, idx):
        text = self._load_text(idx)
        words = text.split()
        text_len = len(words)

        if text_len == 0:
            return None, None

        return self.traindata_gen(words)
        #return self._create_window_samples(words) 先行研究実装, 遅すぎるため自作
    
    def collate_fn(self, l):
        centres, contexts = zip(*l) # l1 -> tuple of centre_ids, l2
        # l1 = [x for x in l1 if x is not None]
        # l2 = [x for x in l2 if x is not None]
        centres = np.hstack(centres)
        contexts = np.concatenate(contexts)
        return torch.from_numpy(centres).long(), torch.from_numpy(contexts).long()


    def __len__(self):
        return self.num_lines
    
    def _load_text(self, idx):
        chunk_number = math.floor(idx / (1.0*self.num_line_per_chunk))
        idx_in_chunk = idx % self.num_line_per_chunk
        with open(self._get_chunk_file_name(chunk_number), "r") as f:
            for i, line in enumerate(f):
                if i == idx_in_chunk:
                    return line.strip()

    def _compute_idx_to_text_word_dict(self):
        idx_to_text_word_tuple = {}
        idx = 0
        for i, text in enumerate(self.texts):
            for j in range(self.text_lengths[i]):
                idx_to_text_word_tuple.update({idx: (i, j)})
                idx += 1

    def _get_chunk_file_name(self, chunk_number):
        return f"/tf/paper/cbow-com/dataset/wikidata/wiki-cleaned.batch_{chunk_number:03d}.txt"
    
    def _count_words_per_line(self):
        text_lengths = None

    def traindata_gen(self, words):
        history = np.zeros((self.window_size*2+1), dtype=np.int32)
        history_idx = 0
        context_ids = []
        centre_ids = []
        for word in words:
            if word is None:
                break
            if word not in self.word_to_id:
                continue
            history[history_idx % (self.window_size * 2 + 1)] = self.word_to_id[word]
            history_idx += 1
            #Check if history countains zero
            if history_idx <= self.window_size * 2:
                continue
            
            context_mask_index = np.ones_like(history, dtype=bool)
            context_mask_index[(history_idx-1 - self.window_size + self.window_size*2+1) % (self.window_size*2+1)] = False
            #centre_ids.append(history[(history_idx- 1 - self.window_size + self.window_size*2+1) % (self.window_size*2+1)])
            centre_ids.append(history[~context_mask_index]) # using ~ to reverse bool
            context_ids.append(history[context_mask_index])
        
        centre_ids = np.array(centre_ids, dtype=np.int32).squeeze(1)
        context_ids = np.array(context_ids, dtype=np.int32)
        return centre_ids, context_ids

    def _create_window_samples(self, words):

        text_len = len(words)
        num_samples = text_len

        training_sequences = np.zeros((num_samples - 2 * self.window_size, 2 * self.window_size), dtype=np.int32)
        centre_words = np.zeros((num_samples - 2 * self.window_size), dtype=np.int32)
        
        #middle_words = random.sample(range(text_len), num_samples)
        middle_words = words[self.window_size : text_len - self.window_size]
        for i, j in enumerate(middle_words):
            #i : enum
            #j ; centre_word
            centre_word = self.window_size + i

            training_sequence = [centre_word + context_word for context_word in range(-self.window_size, self.window_size + 1) if centre_word + context_word  != centre_word]
            training_sequence = [self.word_to_id[words[w]] for w in training_sequence]
            training_sequences[i] = np.array(training_sequence)
            centre_word = words[centre_word]
            centre_words[i] = np.array(self.word_to_id[centre_word])
        
        """
        training_sequencesには, 周辺単語
        centre_wordsには, 中心単語が含まれる
        ...はず
        """
        return training_sequences, centre_words

def _generate_texts(corpus_path_format : str, num_docs: int):
    """
    corpus_path: globで取得するためのフォーマット文字列
    num_docs: データセットに何行のデータが含まれているか
    num_line_per_chunk: 分割時, chunk毎に何行保存するかを指定する.
    注) gensimのWikiCorpusを用いて分割しているので, 1行=1記事扱い.
        尚, wiki-cleaned.txtには5772938行含まれている.
    """
    corpus_path = "/tf/paper/cbow-com/dataset/wikidata/wiki-cleaned.batch_*.txt"
    filename_list = glob.glob(corpus_path)
    random.shuffle(filename_list) #ランダム化

    for filename in filename_list:
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if num_docs is not None and i > num_docs - 1:
                    break
                yield line

def _generate_vocab(corpus_path_format : str, num_line_per_chunk: int):
    tmp = "to_be_implemented"
    return tmp

def _load_texts(corpus_path_format:str , num_line_per_chunk:int):
    """
    _load_texts関数
    ファイルを開き, 記事毎に単語をリスト化して返す
    もちろん一気に読んだら爆発するので, コーパスファイルを分割した上でロードすること
    """
    texts = []
    filename_list = glob.glob('/tf/paper/cbow-com/dataset/wikidata/wiki-cleaned.batch_*.txt')
    filename_list.sort()
    for filename in filename_list:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                print(line)
                texts.append(line)
                if num_line_per_chunk is not None and len(texts) > num_line_per_chunk:
                    break

    return texts

def test():
    vocab_list = []
    with open('/tf/paper/cbow-com/utils/wiki-cleaned.nostopword.100000.vocab') as f:
        for line in f:
            line = line.replace("\n", "")
            vocab_list.append(line)
            
    cbowdataset = CBOWDataset(
        path="/tf/paper/cbow-com/dataset/wikidata/wiki-cleaned.txt",
        num_lines=5772938,
        num_line_per_chunk=50000,
        window_size=5,
        word_vocab=vocab_list,
    )
    data_loader = DataLoader(
        dataset=cbowdataset,
        batch_size=32,
        num_workers=16,
        shuffle=False,
        collate_fn=cbowdataset.collate_fn
    )
    for batch in data_loader:
        print(f"centre_id shape: {batch[0].shape}, context_ids shape{batch[1].shape}")

if __name__ == "__main__":
    test()
