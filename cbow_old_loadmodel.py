import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
from utils import cos_similarity, load_dict, most_similar, check_anisotropy
from CustomLoader import CustomDataset
from operator import itemgetter
import pandas as pd
from scipy.stats import spearmanr, rankdata
from ws353benchmark import ws353_dataframe, simlex999_dataframe, check_vocab, get_dataframe
from analogybenchmark import get_benchmark_dataset, get_file_list
args = sys.argv
SIMILAR_MODE = 'positive'
similar_mode_list = ['positive', 'negative']
CALC_MODE = 'sim'
calc_mode_list = ['sim', 'calc']
SCOPE = 5

window_size = None
emb_dim = None
epoch = None
verbose = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
word_to_id, id_to_word = load_dict("/tf/paper/cbow-com/utils/wiki-cleaned.nostopword.100000.vocab")

if word_to_id is None:
    print("Vocabulary file can't be opened, Abort!")
    sys.exit(1)

if len(args) != 4:
    print("num of parameters doesn't match, abort!")
    print(f"example: python3 {args[0]} [window_size] [emb_dim] [epoch]")
    sys.exit(1)
else:
    window_size = int(args[1])
    emb_dim = int(args[2])
    epoch = int(args[3])
print(f"window_size: {window_size}")
print(f"emb_dim: {emb_dim}")
print(f"epoch: {epoch}")

def eval_analogy_benchmark():
    benchmark_datasets = get_benchmark_dataset()
    acc_list = []
    counter = 0
    semantic_len = 0
    semantic_acc_count = 0
    syntactic_len = 0
    syntactic_acc_count = 0
    
    file_list = get_file_list()

    for i, dataset in enumerate(benchmark_datasets):
        data_count = len(dataset)
        acc_count = 0
        for data in dataset:
            ex_words = torch.tensor([data[0], data[1], data[3]], dtype=torch.int32).to(device)
            predict_vector = model.input_emb(torch.tensor(data[0]).to(device)) - model.input_emb(torch.tensor(data[1]).to(device)) + model.input_emb(torch.tensor(data[3]).to(device))
            predict_vector = predict_vector.to(device)
            res = most_similar(model, vocab_size, predict_vector, scope=1, ex_words=ex_words)
            judge = data[2] == res[0][0]
            judge_symbol = "O" if judge else "X"
            if verbose > 1:
                print(f"[{counter+1:05d}:{judge_symbol}]ans: {id_to_word[data[2]]}, pred: {id_to_word[res[0][0]]}, confidence: {res[0][1]}")
            if judge:
                acc_count += 1
            counter += 1
        print(f"Accuracy: {acc_count / data_count}")
        acc_list.append(acc_count/data_count)
        if "gram" in file_list[i]:
            print(f"{file_list[i]}: SYNTACTIC")
            syntactic_len += data_count
            syntactic_acc_count += acc_count
        else:
            print(f"{file_list[i]}: SEMANTIC")
            semantic_len += data_count
            semantic_acc_count += acc_count
    
    print("Word Analogy Benchmark Result")
    for i, acc in enumerate(acc_list):
        print(f"{file_list[i]}. {acc}")

    print(f"Semantic accuracy: {semantic_acc_count/semantic_len}, {semantic_acc_count} / {semantic_len}")
    print(f"Syntactic accuracy: {syntactic_acc_count/syntactic_len}, {syntactic_acc_count} / {syntactic_len}")
    print(f"Overall accuracy: {(semantic_acc_count+syntactic_acc_count) / (semantic_len + syntactic_len)}")

def eval_benchmark(benchmark_type: str):
    df = get_dataframe(benchmark_type)
    if df is None:
        print("Such benchmark type does not exist!")
        return None
    data_length = len(df)
    word1_series = df['word1']
    word2_series = df['word2']
    human_value_series = df['human_value']

    cos_sim = []
    human_value_list = []

    for i in range(data_length):
        word1 = word1_series[i].lower()
        word2 = word2_series[i].lower()
        human_value = human_value_series[i]

        if word1 is None:
            break
        try:
            word1_vector = model.input_emb(torch.tensor(word_to_id[word1]).to(device))
            word2_vector = model.input_emb(torch.tensor(word_to_id[word2]).to(device))
            cos_sim.append(F.cosine_similarity(word1_vector, word2_vector, dim=0).item())
            human_value_list.append(human_value)
        except KeyError:
            continue

    spearman_correlation, _ = spearmanr(cos_sim, human_value_list)
    return spearman_correlation


def calcmode_sim(model, input, vocab_size) -> None:
    input_check = check_command(input)
    if input_check == -1:
        sys.exit(1)
    if input_check != 10:
        return
    try:
        input_word_id = word_to_id[input]
        input_word_id = torch.tensor(input_word_id, dtype=torch.int32).to(device)
    except KeyError:
        print(f"word {input} is not in the dict !")
        return
    input_word_emb = model.input_emb(input_word_id)
    similar_words = most_similar(model, vocab_size, input_word_emb, scope=SCOPE, mode=SIMILAR_MODE, ex_words=input_word_id)
    for word in similar_words:
        print(f"{id_to_word[word[0]]} : {word[1]}")
    
    return

def calcmode_calc(model, input, vocab_size) -> None:
    input_check = check_command(input)
    if input_check == -1:
        sys.exit(1)
    print("a")
    if input_check != 10:
        return
    print("b")
    input_words = input.split()
    print(input_words)
    if len(input_words) != 3:
        print("input 3 words!")
        return
    input_words_id = []
    for input_word in input_words:
        try:
            tmp = word_to_id[input_word]
            input_words_id.append(tmp)
        except KeyError:
            print(f"word {input_word} is not in the dict!")
            return
    
    input_words_id = torch.tensor(input_words_id, dtype=torch.int)

    input_words_emb = model.input_emb(input_words_id)

    predicted_emb = input_words_emb[0] - input_words_emb[1] + input_words_emb[2]

    similar_words = most_similar(model, vocab_size, predicted_emb, scope=SCOPE, mode=SIMILAR_MODE, ex_words=input_words_id)

    for word in similar_words:
        print(word)
        print(f"{id_to_word[word[0]]}: {word[1]}")

    return None

def check_command(input) -> int:
    if input == "2231081":
        print("bye")
        return -1
    
    cmd = input.split('=')
    if len(cmd) != 2:
        return 10

    if cmd[0] == 'similar_mode':
        if cmd[1] in similar_mode_list:
            global SIMILAR_MODE
            SIMILAR_MODE = cmd[1]
            print(f"similar_mode is changed: {SIMILAR_MODE}")
            return 1
        else:
            return 0
    if cmd[0] == 'calc_mode':
        if cmd[1] in calc_mode_list:
            global CALC_MODE
            CALC_MODE = cmd[1]
            print(f"calc_mode is changed: {CALC_MODE}")
            return 1
        else:
            return 0
    if cmd[0] == 'scope':
        global SCOPE
        SCOPE = int(cmd[1])
        print(f"SCOPE is changed: {SCOPE}")
        return 1

    return 10

class CBOWNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, weights=None, padding_idx=0):
        super(CBOWNet, self).__init__()
        self.weights = weights
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.input_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=self.padding_idx)#パディングインデックスを追加するためvocab_sizeは実際の語彙数より+1となる
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        #Attention Word Embeddingでは、input_embとoutput_embは一致させるが、今のところは別とする

        #ネガティブサンプリングのやつ. 今回の実装ではすでに0.75乗してあるので無効化する
        # if self.weights is not None:
        #     wf = np.power(self.weights, 0.75)
        #     wf /= np.sum(wf)
        #     self.weights = torch.FloatTensor(wf)
        
    def forward(self, context_ids, centre_id, neg_ids):

        #1. 周辺単語のベクトルを取得
        context_embedding = self.input_emb(context_ids) # shape: (batch_size, window_size*2, emb_dim)
        
        #2. ウィンドウ内の周辺単語のベクトルの平均をとる、これが中間層への入力となる
        context_embedding = torch.mean(context_embedding, dim=1) # shape: (batch_size, emb_dim)
        
        #3. 出力層側での正例の単語ベクトルを取得
        positive_sample = self.output_emb(centre_id) # shape: (batch_size, emb_dim)
        
        #4. 正例に関するネットワークのそのままの出力(=スコア)を計算
        positive_score = torch.sum(context_embedding * positive_sample, dim=1) # shape: (batch_size)
        
        #5. 正例スコアをシグモイド関数(-1, 1)に通した後logで尤度化(-inf, 0)
        positive_loss = F.logsigmoid(positive_score) #shape: (batch_size)

        #6. 負例単語に関する出力側の単語ベクトルを取得
        negative_embedding = self.output_emb(neg_ids) # shape: (batch_size, num_negative_samples, emb_dim)
        
        #7. 正例についてのコンテキストベクトルのサイズを、負例データとの行列積が可能になるように調整
        context_embedding = context_embedding.unsqueeze(1) # shape: (batch_size, 1, emb_dim)

        #8. 負例に関するネットワークのそのままの出力(=スコア)を計算
        negative_score = torch.bmm(context_embedding, negative_embedding.transpose(1, 2)).squeeze(1) # shape: (batch_size, num_negatice_samples)
        
        #9. 負例スコアをシグモイド関数(-1, 1)に通した後logで尤度化(-inf, 0)
        negative_loss = F.logsigmoid(-negative_score).sum(dim=1) #shape: (batch_size)

        #10. 正例での誤差と負例での誤差を足し、最小化問題のためマイナス符号をつけ, バッチ方向で平均を取りスカラに変換、戻り値とする。
        return -(positive_loss + negative_loss).mean() #バッチ方向で平均

vocab_size = 100000
num_negative_samples = 5
model = CBOWNet(vocab_size+1, emb_dim=emb_dim)
model.to(device)
model.load_state_dict(torch.load(f"/tf/paper/cbow-com/models/dim_{emb_dim}/window_{window_size}/model_CBOW_AdamW_default_batch_32768_init_normalized_negs_5_epoch_{epoch}.pth"))

model.eval()
with torch.no_grad():

    norms = torch.linalg.norm(model.input_emb.weight.data, dim=1)
    print(norms.mean().item(), norms.std().item())

    print(check_anisotropy(model, vocab_size, size=10000))
    
    analogy_start = time.time()
    eval_analogy_benchmark()

    analogy_end = time.time()
    print(f"duration; {analogy_end - analogy_start}")

    #print("normalizing weight")
    #model.input_emb.weight.data = F.normalize(model.input_emb.weight.data, p=2, dim=1)

    #norms = torch.linalg.norm(model.input_emb.weight.data, dim=1)
    print(norms.mean().item(), norms.std().item())

    ws353_spearman_correlation = eval_benchmark('ws353')
    print(f"WS353 Spearman correlation: {ws353_spearman_correlation}")

    simlex999_spearman_correlation = eval_benchmark('simlex999')
    print(f"SimLex-999 Spearman correlation: {simlex999_spearman_correlation}")

    men_spearman_correlation = eval_benchmark('men')
    print(f"MEN Spearman correlation: {men_spearman_correlation}")

    ws353r_spearman_correlation = eval_benchmark('ws353r')
    print(f"WS353 Relatedness Spearman correlation: {ws353r_spearman_correlation}")

    ws353s_spearman_correlation = eval_benchmark('ws353s')
    print(f"WS353 Similarity Spearman correlation: {ws353s_spearman_correlation}")

    while True:
        if CALC_MODE == 'sim':
            print("input word to find most similar words.")
        
        elif CALC_MODE == 'calc':
            print("input three words to calc vector (word1 - word2 + word3)")

        word = input()
        
        if CALC_MODE == 'sim':
            calcmode_sim(model, word, vocab_size)
            continue
        
        if CALC_MODE == 'calc':
            calcmode_calc(model, word, vocab_size)
            continue
