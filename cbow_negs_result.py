import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import time
import sys
import glob
import gc
from functools import partial
from tqdm import tqdm as std_tqdm
from utils import cos_similarity, load_dict, glove_weight_function, most_similar
from CustomLoader import CustomDataset#, WikiDataset
from scipy import sparse
import pandas as pd
from scipy.stats import spearmanr, rankdata
from ws353benchmark import get_dataframe
from analogybenchmark import get_benchmark_dataset, get_file_list

device = "cuda" if torch.cuda.is_available else "cpu"
args = sys.argv
verbose = 1
print("Model Loader, CBOW with Cooccurrence Matrix, negative sampling")
if len(args) <= 0:
    print("no parameters are set, use default settings")
    window_size = 5
    emb_dim = 300
    epoch = 4
elif len(args) != 4:
    print("num of parameters doesn't match, abort!")
    print(f"example: python3 {args[0]} [window_size] [emb_dim] [epoch]")
    sys.exit(1)
else:
    window_size = int(args[1])
    emb_dim = int(args[2])
    epoch = int(args[3])

word_to_id, id_to_word = load_dict("/tf/paper/cbow-com/utils/wiki-cleaned.nostopword.100000.vocab")

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
    semantic_accuracy = semantic_acc_count / semantic_len
    syntactic_accuracy = syntactic_acc_count / syntactic_len
    overall_accuracy = (semantic_acc_count + syntactic_acc_count) / (semantic_len + syntactic_len)
    print(f"Semantic accuracy: {semantic_accuracy}, {semantic_acc_count} / {semantic_len}")
    print(f"Syntactic accuracy: {syntactic_accuracy}, {syntactic_acc_count} / {syntactic_len}")
    print(f"Overall accuracy: {overall_accuracy}")

    return semantic_accuracy, syntactic_accuracy, overall_accuracy


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
            cos_sim.append(cos_similarity(word1_vector, word2_vector).item())
            human_value_list.append(human_value)
        except KeyError:
            continue
    
    #cos_sim = rankdata(cos_sim)
    #human_value_list = rankdata(human_value_list)

    spearman_correlation, _ = spearmanr(cos_sim, human_value_list)
    return spearman_correlation

class CBOWNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, weights=None, padding_idx=0):
        super(CBOWNet, self).__init__()
        self.weights = weights
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        #self.input_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=self.padding_idx)#パディングインデックスを追加するためvocab_sizeは実際の語彙数より+1となる
        self.input_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=self.padding_idx)
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
        #テスト: EmbeddingBagによって自動的に平均化してみる
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
        negative_score = torch.bmm(context_embedding, negative_embedding.transpose(1, 2)).squeeze(1) # shape: (batch_size, num_negative_samples)

        #9. 負例スコアをシグモイド関数(0, 1)に通した後logで尤度化(-inf, 0)
        negative_loss = F.logsigmoid(-negative_score).sum(dim=1) #shape: (batch_size)

        #10. 正例での誤差と負例での誤差を足し、最小化問題のためマイナス符号をつけ, バッチ方向で平均を取りスカラに変換、戻り値とする。
        loss = -(positive_loss + negative_loss).mean()
        return loss #バッチ方向で平均


vocab_size = 100000
batch_size = 32768
num_negative_samples = 5
model = CBOWNet(vocab_size+1, emb_dim=emb_dim)

model.load_state_dict(torch.load(f"/tf/paper/cbow-com/models/dim_{emb_dim}/window_{window_size}/model_CBOW_AdamW_default_batch_32768_init_normalized_negs_{num_negative_samples}_epoch_{epoch}.pth"))
model.to(device)
model.eval()

with torch.no_grad():
    with open("./experiment_analogy_result.txt", 'a') as file:
        analogy_result = eval_analogy_benchmark()
    
        print(f"Semantic: {analogy_result[0]}, Syntactic: {analogy_result[1]}, Overall: {analogy_result[2]}")
        file.write(f"CBOW {window_size} {emb_dim} {epoch} : {analogy_result[0]}, {analogy_result[1]}, {analogy_result[2]}\n")
    sys.exit(1)

    with open("./experiment_benchmark_result.txt", 'a') as file:

        ws353_spearman_correlation = eval_benchmark('ws353')
        print(f"WS353 Spearman correlation: {ws353_spearman_correlation}")

        simlex999_spearman_correlation = eval_benchmark('simlex999')
        print(f"SimLex-999 Spearman correlation: {simlex999_spearman_correlation}")
        
        men_spearman_correlation = eval_benchmark('men')
        print(f"MEN Spearman correlation: {men_spearman_correlation}")

        file.write(f"CBOW {window_size} {emb_dim} {epoch} : {ws353_spearman_correlation}, {simlex999_spearman_correlation}, {men_spearman_correlation}\n")

