import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataloader import default_collate

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

# class WikiDataset(Dataset):
#     def __init__(self, corpus_path, vocab_path, window_size):
#         self.window_size = window_size
#         self.dict = {}
#         self.data = []
#         self.pairs = []
#         id_counter = 1
#         with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
#             for line in vocab_file:
#                 self.dict[line] = id_counter
#                 id_counter += 1
        
#         with open(corpus_path, 'r', encoding='utf-8') as corpus:
#             #一行ずつ読み込む
#             for line in corpus:
#                 # 普通にTokenize
#                 for word in line.split():
#                     #vocabに含まれているものだけデータとして取り込む
#                     for word in self.dict:
#                         self.data.append(self.dict[word])
        
#         self.pairs = []
#         for i, center_word in enumerate(self.data):
#             start = i - self.window_size
#             end = min(len(self.data), i + self.window_size + 1)
#             for j in range(start, end):
#                 if i == j:
#                     continue
#                 self.pairs.append((center_word, context_word))
        
#         def __len__(self):
#             return len(self.pairs)
        
#         def __getitem__(self, idx):
#             center, context = self.pairs[idx]
        
#         return torch.tensor(center), torch.tensor(context)

# class ComatDataset(Dataset):
#     def __init__():
#         return
    
#     def __len__():
#         return
    
#     def __getitem__():
#         return
#collate_fn: データセットから取り出されたデータのリストを入力にとる
#CBOW ONLY, NOT FOR SKIP-GRAM

def tokenize(text):
    return text.split()

class CBOWDataset(Dataset):
    def __init__(self, path, window_size, word_vocab):
        self.window_size = window_size


    def __getitem__(self, idx):

        """
        Gensimで分割したwikipediaのデータは, article一つ毎に一回改行している
        一行ロード = 記事一個ロードする、ということ
        """
        text = self._load_text(idx)
        words = text.split()
        text_len = len(words)

        if text_len == 0:
            return None