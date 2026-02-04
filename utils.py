import numpy as np
import torch
import torch.nn as nn
import collections

device = "cuda" if torch.cuda.is_available() else "cpu"

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


"""
    cos_similarity_gpu
    input:
        x: all embedding vectors, x.dim must be 2
        y: an embedding vector to check, y.dim must be 0
    
    return: cosine similarity between each x tensor and y tensor
"""
def cos_similarity_bag(x: torch.tensor, y: torch.tensor, eps=1e-8):
    if y.dim() != 1:
        return None
    x_numerator = torch.sum(torch.pow(x, 2), dim=1, keepdim=True)
    x_numerator = torch.sqrt(x_numerator) + eps
    nx = x / x_numerator

    y_numerator = torch.sum(torch.pow(y, 2), dim=0, keepdim=True)
    y_numerator = torch.sqrt(y_numerator) + eps
    ny = y / y_numerator
    ny = ny.expand(x.size(0), -1)

    return torch.sum((nx * ny), dim=1) 

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
# def glove_weight_function(x, x_max=100):
#     return np.clip(np.div(x, x_max), 1e-8, 1)

def glove_weight_function(x, x_max=100):
    return torch.clip(torch.div(x, x_max), 1e-8, 1)

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

def load_vocab(filename):
    vocab_list = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            vocab_list.append(line)
    return vocab_list

def check_anisotropy(model, vocab_size, size=10000):
    rng = np.random.default_rng(0)
    

    random_1 = rng.integers(low=0, high=vocab_size-1, size=size)
    random_2 = rng.integers(low=0, high=vocab_size-1, size=size)

    random_1_emb = model.input_emb(torch.tensor(random_1).to(device))
    random_2_emb = model.input_emb(torch.tensor(random_2).to(device))

    cos_sim = torch.nn.functional.cosine_similarity(random_1_emb, random_2_emb)
    cos_sim_ave = torch.mean(cos_sim)
    return cos_sim_ave

def most_similar(model, vocab_size, word_emb, scope=5, mode="positive", ex_words=None):

    cos_sim_list = -np.ones((scope))
    cos_sim_wordlist = [""*scope]
    counter = 1
    min_cos_sim = -1.0

    one_float_tensor = torch.tensor(1.0, dtype=torch.float32)
    emb_idx = torch.arange(vocab_size+1).to(device)
    #cos_sim = cos_similarity_bag(model.input_emb(emb_idx), word_emb)
    cos_sim = torch.nn.functional.cosine_similarity(model.input_emb(emb_idx), word_emb)
    if ex_words is not None:
        cos_sim[ex_words] = -1
    
    cos_sim, indices = torch.sort(cos_sim, dim=0, descending=True)
    res = []
    for i in range(scope):
        res.append([indices[i].item(), cos_sim[i].item()])

    return res

    for i in range(1, vocab_size):
        if i in ex_words:
            continue
        cos_sim = cos_similarity(model.input_emb(torch.tensor(i)), word_emb)
        if cos_sim > cos_sim_list.min():
            #最小値よりも大きければ, 書き換え
            cos_sim_list[np.argmin(cos_sim_list)] = cos_sim
            cos_sim_wordlist[np.argmin(cos_sim_list)] = i
    
    if mode == 'positive':
        cos_sim_list = np.sort(cos_sim_list)[::-1]
    else:
        cos_sim_list = np.sort(cos_sim_list)

    res = []
    for i in range(scope):
        res.append((cos_sim_list[i], cos_sim_wordlist[i]))
    res.sort(key=lambda x: x[1], reverse=True if mode == 'positive' else False)
    return res

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
