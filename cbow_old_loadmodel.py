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
from utils import cos_similarity
from utils import load_dict
from WikiLoader import CustomDataset
import pandas as pd
from scipy.stats import spearmanr, rankdata
from ws353benchmark import ws353_dataframe, simlex999_dataframe

args = sys.argv

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

emb_dim = 300
model = CBOWNet(100001, emb_dim=emb_dim)
word_to_id, id_to_word = load_dict("/tf/paper/cbow-com/wiki-cleaned.nostopword.100000.vocab")

if word_to_id is None:
    print("Vocabulary file can't be opened, Abort!")
    sys.exit(1)

model.load_state_dict(torch.load("/tf/paper/cbow-com/models/dim_300/model_AdamW_default_negs_epoch_1_120.pth"))

model.eval()
with torch.no_grad():

    pred_paris = model.input_emb.weight[1769] - model.input_emb.weight[463] + model.input_emb.weight[362]
    ans_paris = model.input_emb.weight[608]

    print("predicted Paris :"); print(pred_paris)
    print("True      Paris :"); print(ans_paris)
    print("cos similarity: "); print(cos_similarity(pred_paris, ans_paris))

    pred_father = model.input_emb.weight[500] - model.input_emb.weight[656] + model.input_emb.weight[277]
    ans_father = model.input_emb.weight[316]
    print(f"cos similarity of pred and ans father: "); print(cos_similarity(pred_father, ans_father))

    pred_queen = model.input_emb.weight[251] - model.input_emb.weight[277] + model.input_emb.weight[656]
    ans_queen = model.input_emb.weight[815]
    print(f"cos similarity of pred and ans queen: "); print(cos_similarity(pred_queen, ans_queen))

    said_minus_say = model.input_emb.weight[276] - model.input_emb.weight[1970] #said - say  = called - call となったりしないか？
    called_minus_call = model.input_emb.weight[175] - model.input_emb.weight[1197] 
    print("cos similarity of pred and ans past :"); print(cos_similarity(said_minus_say, called_minus_call)) # コサイン類似度が近いのでおそらくsaid_minus_sayは動詞を過去形にするベクトル？

    df_ws353 = ws353_dataframe()
    word1_series = df_ws353['word1']
    word2_series = df_ws353['word2']
    human_value_series = df_ws353['human_value']
    data_length = 353 # ws353

    ws353_cos_sim = []
    ws353_human_value = []
    for i in range(data_length):
        word1 = word1_series[i].lower()
        word2 = word2_series[i].lower()
        human_value = human_value_series[i]
        if word1 is None:
            break
        
        word1_vector = model.input_emb(torch.tensor(word_to_id[word1]))
        word2_vector = model.input_emb(torch.tensor(word_to_id[word2]))
        ws353_cos_sim.append(cos_similarity(word1_vector, word2_vector).item())
        ws353_human_value.append(human_value)
    
    ws353_cos_sim = rankdata(ws353_cos_sim)
    ws353_human_value = rankdata(ws353_human_value)

    ws353_spearman_correlation, pvalue = spearmanr(ws353_cos_sim, ws353_human_value)
    print(ws353_cos_sim)
    print(ws353_human_value)
    print(f"WS353 Spearman relation: {ws353_spearman_correlation}")

    df_simlex999 = simlex999_dataframe()
    data_length_simlex999 = 999
    word1_series_simlex999 = df_simlex999['word1']
    word2_series_simlex999 = df_simlex999['word2']
    human_value_series_simlex999 = df_simlex999['human_value']
    
    simlex999_cos_sim = []
    simlex999_human_value = []
    for i in range(data_length_simlex999):
        word1 = word1_series_simlex999[i].lower()
        word2 = word2_series_simlex999[i].lower()
        human_value = human_value_series_simlex999[i]
        if word1 is None:
            break
        
        word1_vector = model.input_emb(torch.tensor(word_to_id[word1]))
        word2_vector = model.input_emb(torch.tensor(word_to_id[word2]))
        simlex999_cos_sim.append(cos_similarity(word1_vector, word2_vector).item())
        simlex999_human_value.append(human_value)
    
    simlex999_cos_sim = rankdata(simlex999_cos_sim)
    simlex999_human_value = rankdata(simlex999_human_value)

    simlex999_spearman_correlation, pvalue = spearmanr(simlex999_cos_sim, simlex999_human_value)
    print(f"SimLex-999 Spearman relation: {simlex999_spearman_correlation}")

    while True:
        print("input words. format: word1, word2")

        words = input()

        words = words.split(" ")

        id_1 = words[0]

        if id_1 == "2231081":
            print("BYE")
            break

        try:
            id_1 = word_to_id[id_1]
        except KeyError:
            print("that word is not in dict!")
            continue
        
        print("input second word:")
        id_2 = words[1]

        try:
            id_2 = word_to_id[id_2]
        except KeyError:
            print("that word is not in dict!")
            continue

        print(cos_similarity(model.input_emb.weight[id_1], model.input_emb.weight[id_2]))
