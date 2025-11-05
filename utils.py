#Coded by Kouki TAKASHIMA, Chiba Institute of Technology
#utils.py

import numpy as np
import torch
import torch.nn as nn
import collections

def preprocess(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace('.', ' .')
    words = sentence.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = torch.asarray([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size: -window_size]
    contexts = []

    for idx in range(window_size, len(corpus)- window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    
    return torch.asarray(contexts), torch.asarray(target)

def cos_similarity(x, y, eps=1e-8):
    nx = x / (torch.sqrt(torch.sum(x ** 2)) + eps)
    ny = y / (torch.sqrt(torch.sum(y ** 2)) + eps)
    return torch.dot(nx, ny)

def generate_word_probability(corpus, vocab_size):
    counts = collections.Counter()

    for word_id in corpus:
        counts[word_id] += 1
    
    word_prob = torch.zeros(vocab_size)
    for i in range(vocab_size):
        word_prob[i] = counts[i]
    
    word_prob = torch.pow(word_prob, 0.75)
    word_prob /= torch.sum(word_prob)

    return word_prob

def generate_negative_samples(corpus, word_prob, centre_word_idx):
    word_prob[centre_word_idx] = 0
    word_prob /= (1 / torch.sum(word_prob))

# x: torch.tensor, shape: [batch_size]
def glove_weight_function(x, x_max=100):
    return np.clip(np.div(x, x_max), 1e-8, 1)

def generate_co_occurrence_matrix(corpus, vocab_size, window_size=1):

    co_matrix = torch.zeros((vocab_size, vocab_size))

    for idx in range(window_size, vocab_size-window_size):
        target_word_id = corpus[idx]
        for offset in range(-window_size, window_size+1):
            if offset == 0:
                continue
            context_word_id = corpus[idx + offset]
            co_matrix[target_word_id][context_word_id] += 1

    return co_matrix


def _init_normal(vocab_size, emb_dim):
    init_weights = torch.from_numpy(np.random.normal(size=(vocab_size, emb_dim),
        loc = 0.0, scale = 0.1).astype(np.float32))

    return init_weights

def load_dict(filename):
    word_to_id = {}
    id_to_word = {}
    counter = 1
    try:
        with open(filename) as f:
            for line in f:
                line = line.replace("\n", "")
                word_to_id[line] = counter
                id_to_word[counter] = line
                counter += 1
    except FileNotFoundError as e:
        print(f"{e}")
        return None, None

    return word_to_id, id_to_word

def most_similar(model, vocab_size, word_emb, scope=5, mode="positive"):

    cos_sim_list = []
    counter = 1
    one_float_tensor = torch.tensor(1.0, dtype=torch.float32)
    other_words = torch.tensor([i for i in range(1, vocab_size)])
    other_word_embs = model.input_emb(other_words)

    for other_word_emb in other_word_embs:
        cos_sim = cos_similarity(other_word_emb, word_emb)
        if cos_sim >= one_float_tensor:
            #コサイン類似度が1なら流石に同一単語
            counter += 1
            continue
        cos_sim_list.append([counter, cos_sim])
        counter += 1
    cos_sim_list.sort(key=lambda x: x[1], reverse=True if mode == 'positive' else False)
    return cos_sim_list[:scope]
