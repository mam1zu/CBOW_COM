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
from utils import cos_similarity, load_dict, most_similar
from CustomLoader import CustomDataset#, WikiDataset
from scipy import sparse
import pandas as pd
from scipy.stats import spearmanr, rankdata
from ws353benchmark import get_dataframe
from analogybenchmark import get_benchmark_dataset, get_file_list
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available else "cpu"
verbose = 1

args = sys.argv
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


class CBOWNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, weights=None, padding_idx=0):
        super(CBOWNet, self).__init__()
        self.weights = weights
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.input_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=self.padding_idx)#パディングインデックスを追加するためvocab_sizeは実際の語彙数より+1となる
        #self.input_emb = nn.EmbeddingBag(vocab_size, emb_dim, mode='mean', padding_idx=self.padding_idx)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        #Attention Word Embeddingでは、input_embとoutput_embは一致させるが、今のところは別とする
        self.eps = 1e-8
        self.x_max = 100
        self.alpha = 0.75
        self.comat_rate = 1
        #ネガティブサンプリングのやつ. 今回の実装ではすでに0.75乗してあるので無効化する
        # if self.weights is not None:
        #     wf = np.power(self.weights, 0.75)
        #     wf /= np.sum(wf)
        #     self.weights = torch.FloatTensor(wf)
    
    # glove-weighting
    def forward_proposed(self, context_ids, centre_id, neg_ids, comat_data):
        context_embedding = self.input_emb(context_ids) # shape: [batch_size, window_size*2, emb_dim]
        comat_data = glove_weight_function(comat_data)
        comat_sum = torch.sum(comat_data, dim=1, keepdim=True)
        comat_data = comat_data / (comat_sum + self.eps)
        comat_data = comat_data.unsqueeze(2)

        context_embedding = context_embedding * comat_data.expand(-1, -1, emb_dim)
        context_embedding = torch.sum(context_embedding, dim=1)

        positive_sample = self.output_emb(centre_id)
        positive_score = torch.sum(context_embedding * positive_sample, dim=1)
        positive_loss = F.logsigmoid(positive_score)

        negative_embedding = self.output_emb(neg_ids)
        context_embedding = context_embedding.unsqueeze(1)

        negative_score = torch.bmm(context_embedding, negative_embedding.transpose(1, 2)).squeeze(1)
        negative_loss = F.logsigmoid(-negative_score).sum(dim=1)
        loss = -(positive_loss + negative_loss).mean()
        return loss
    #inverse-log weighting
    def forward_inverselog(self, context_ids, centre_id, neg_ids, comat_data):
            context_embedding = self.input_emb(context_ids) # shape: [batch_size, window_size*2, emb_dim]
            #comat_row_idx = centre_id[:, torch.newaxis]

            #共起行列(密行列)は, サイズを有効活用するためにインデックスを圧縮している
            #例: 単語ID1番の共起情報は, comat[0]に保存されている, インデックスの調整に注意すること

            #comat_data = comat[comat_row_idx - 1, context_ids - 1].float() #shape; [batch_size, window_size*2]
            #comat_data = torch.div(-comat_data, self.x_max) # Glove weight function phase 1
            #comat_data = torch.pow(comat_data, self.alpha) # GloVe weight function phase 2
            comat_data = torch.reciprocal(comat_data) #<-ここで逆数をとってる
            comat_sum = torch.sum(comat_data, dim=1, keepdim=True) #shape: [batch_size, 1]
            comat_data = comat_data / (comat_sum + self.eps) # shape: [batch_size, window_size*2]
            comat_data = comat_data.unsqueeze(2) # shape: [batch_size, window_size*2, 1]

            context_embedding_comat = context_embedding * comat_data.expand(-1, -1, emb_dim) #shape: [batch_size, window_size*2, emb_dim]
            context_embedding_comat = torch.sum(context_embedding_comat, dim=1) #shape: [batch_size, emb_dim]

            context_embedding = torch.mean(context_embedding, dim=1)
            context_embedding = (1.0 - self.comat_rate) * context_embedding + self.comat_rate * context_embedding_comat
            # 出力層側での正例の単語ベクトルを取得
            positive_sample = self.output_emb(centre_id)

            # 正例に関するネットワークのそのままの出力(=スコア)を計算
            positive_score = torch.sum(context_embedding * positive_sample, dim=1) # shape: (batch_size)

            # 正例スコアをシグモイド関数(0, 1)に通した後logで尤度化(-inf, 0)
            positive_loss = F.logsigmoid(positive_score) #shape: (batch_size)

            # 負例単語に関する出力側の単語ベクトルを取得
            negative_embedding = self.output_emb(neg_ids) # shape: (batch_size, num_negative_samples, emb_dim)
            
            # 正例についてのコンテキストベクトルのサイズを, 負例データとの行列積が可能になるように調整
            context_embedding = context_embedding.unsqueeze(1) # shape: (batch_size, 1, emb_dim)

            # 負例に関するネットワークのそのままの出力(=スコア)を計算
            negative_score = torch.bmm(context_embedding, negative_embedding.transpose(1, 2)).squeeze(1) # shape: (batch_size, num_negative_samples)
            
            # 負例スコアのシグモイド関数(0, 1)に通した後logで尤度化(-inf, 0)
            negative_loss = F.logsigmoid(-negative_score).sum(dim=1) #shape: (batch_size)
            loss = -(positive_loss + negative_loss).mean() #バッチ方向で平均
            return loss
    # log weighting
    def forward(self, context_ids, centre_id, neg_ids, comat_data):
        context_embedding = self.input_emb(context_ids) # shape: [batch_size, window_size*2, emb_dim]
        #comat_row_idx = centre_id[:, torch.newaxis]

        #共起行列(密行列)は, サイズを有効活用するためにインデックスを圧縮している
        #例: 単語ID1番の共起情報は, comat[0]に保存されている, インデックスの調整に注意すること

        #comat_data = comat[comat_row_idx - 1, context_ids - 1].float() #shape; [batch_size, window_size*2]
        #comat_data = torch.div(-comat_data, self.x_max) # Glove weight function phase 1
        #comat_data = torch.pow(comat_data, self.alpha) # GloVe weight function phase 2
        comat_sum = torch.sum(comat_data, dim=1, keepdim=True) #shape: [batch_size, 1]
        comat_data = comat_data / (comat_sum + self.eps) # shape: [batch_size, window_size*2]
        comat_data = comat_data.unsqueeze(2) # shape: [batch_size, window_size*2, 1]

        context_embedding_comat = context_embedding * comat_data.expand(-1, -1, emb_dim) #shape: [batch_size, window_size*2, emb_dim]
        context_embedding_comat = torch.sum(context_embedding_comat, dim=1) #shape: [batch_size, emb_dim]

        context_embedding = torch.mean(context_embedding, dim=1)
        context_embedding = (1.0 - self.comat_rate) * context_embedding + self.comat_rate * context_embedding_comat
        # 出力層側での正例の単語ベクトルを取得
        positive_sample = self.output_emb(centre_id)

        # 正例に関するネットワークのそのままの出力(=スコア)を計算
        positive_score = torch.sum(context_embedding * positive_sample, dim=1) # shape: (batch_size)

        # 正例スコアをシグモイド関数(0, 1)に通した後logで尤度化(-inf, 0)
        positive_loss = F.logsigmoid(positive_score) #shape: (batch_size)

        # 負例単語に関する出力側の単語ベクトルを取得
        negative_embedding = self.output_emb(neg_ids) # shape: (batch_size, num_negative_samples, emb_dim)
        
        # 正例についてのコンテキストベクトルのサイズを, 負例データとの行列積が可能になるように調整
        context_embedding = context_embedding.unsqueeze(1) # shape: (batch_size, 1, emb_dim)

        # 負例に関するネットワークのそのままの出力(=スコア)を計算
        negative_score = torch.bmm(context_embedding, negative_embedding.transpose(1, 2)).squeeze(1) # shape: (batch_size, num_negative_samples)
        
        # 負例スコアのシグモイド関数(0, 1)に通した後logで尤度化(-inf, 0)
        negative_loss = F.logsigmoid(-negative_score).sum(dim=1) #shape: (batch_size)
        loss = -(positive_loss + negative_loss).mean() #バッチ方向で平均
        return loss
        
        
    def forward_old(self, context_ids: torch.tensor, centre_id: torch.tensor, neg_ids: torch.tensor, comat_data: torch.tensor):
        
        # context_ids: [batch_size, window_size*2], dtype: int64
        # centre_id  : [batch_size], dtype: int64
        # neg_ids    : [batch_size, num_negative_samples, window_size*2], dtype: int64
        # comat_data : [batch_size, window_size*2], dtype: float32

        #1. 周辺単語のベクトルを取得
        context_embedding = self.input_emb(context_ids) # shape: (batch_size, window_size*2, emb_dim)
        #print(f"context_embedding.shape: {context_embedding.shape}")
        #print(f"comat_data.shape: {comat_data.shape}")

        #2. ウィンドウ内の周辺単語のベクトルの平均をとる、これが中間層への入力となる
        #テスト: EmbeddingBagによって自動的に平均化してみる

        # weight_left_singular = self.left_singular_table(centre_id) #shaoe; [batch_size, k_dim]
        # weight_left_singular = weight_left_singular.unsqueeze(1) #shape: [batch_size, 1, k_dim]
        # weight_right = self.right_table(context_ids) #shape: [batch_size, window_size*2, k_dim]

        comat_sum = torch.sum(comat_data, dim=1, keepdim=True) #shape: [batch_size, 1]
        #print(f"comat_sum.shape: {comat_sum.shape}")
        comat_data = comat_data / comat_sum #shape: [batch_size, window_size*2]
        #print(f"comat_data divided by sum shape: {comat_data.shape}")
        comat_data = comat_data.unsqueeze(2) #shape: [batch_size, window_size*2, 1]
        #print(f"unsqueezed comat_data.shape: {comat_data.shape}")
        context_embedding_comat = context_embedding * comat_data.expand(-1, -1, emb_dim) #shape: [batch_size, window_size*2, emb_dim]
        context_embedding_comat = torch.sum(context_embedding, dim=1) #shape: [batch_size, emb_dim]
        context_embedding = (1.0 - self.comat_rate)*context_embedding + self.comat_rate * context_embedding_comat

        #weight_right.transpose(1, 2) shape: [batch_size, k_dim, window_size*2]
        # gloveweight = torch.matmul(weight_left_singular, weight_right.transpose(1, 2)) #shape: [batch_size, 1, window_size*2]
        # gloveweight = gloveweight.transpose(1, 2) #shape:; [batch_size, window_size*2, 1]
        # gloveweight = torch.clip(gloveweight, min=1e-3, max=1)

        # gloveweight_sum = torch.sum(gloveweight, dim=1, keepdim=True) #shape: [batch_size, 1, 1]
        # gloveweight = gloveweight / gloveweight_sum #正規化

        # context_embedding = context_embedding * gloveweight.expand(-1, -1, self.emb_dim)
        # context_embedding = torch.sum(context_embedding, dim=1) #shape: [batch_size, emb_dim]

        #gloveweight.repeat(1, 1, self.emb_dim) #shape: [batch_size, window_size*2, emb_dim] <- context_embeddingと同形
        #context_embedding = context_embedding * gloveweight.expand(-1, -1, self.emb_dim) # Hadamard
        #weight_sum = torch.sum(gloveweight, dim=1), #shape: [batch_size, ]
        #context_embedding = context_embedding / weight_sum
        #context_embedding = torch.mean(context_embedding, dim=1)#shape: [batch_size, emb_dim]
        #gloveweight = torch.sum(gloveweight, dim=1).squeeze(1)
        #torch.sum(gloveweight, dim=1).squeeze(), shape: [batch_size]
        #context_embedding = torch.div(context_embedding, torch.sum(gloveweight, dim=0)) #shape: [batch_size, emb_dim]

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
        
        #9. 負例スコアをシグモイド関数(-1, 1)に通した後logで尤度化(-inf, 0)
        negative_loss = F.logsigmoid(-negative_score).sum(dim=1) #shape: (batch_size)

        #10. 正例での誤差と負例での誤差を足し、最小化問題のためマイナス符号をつけ, バッチ方向で平均を取りスカラに変換、戻り値とする。
        return -(positive_loss + negative_loss).mean() #バッチ方向で平均

pca = PCA(n_components=2)

num_negative_samples = 5
vocab_size = 100000
batch_size = 32768
model = CBOWNet(vocab_size+1, emb_dim=emb_dim)

word_to_id, id_to_word = load_dict("/tf/paper/cbow-com/utils/wiki-cleaned.nostopword.100000.vocab")

model.load_state_dict(torch.load(f"/tf/paper/cbow-com/models/dim_{emb_dim}/window_{window_size}/model_CBOW_AdamW_default_batch_32768_init_normalized_negs_{num_negative_samples}_epoch_{epoch}.pth"))
#model.to(device)
model.eval()
with torch.no_grad():

    pca.fit(model.input_emb.weight)

    data_pca= pca.transform(model.input_emb.weight)

    fig=plt.figure(figsize=(20,12),facecolor='w')

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    # # 3Dを指定
    # ax = fig.add_subplot(111, projection="3d")

    # # 各軸の設定
    # ax.set_xlabel("x", size=10)
    # ax.set_ylabel("y", size=10)
    # ax.set_zlabel("z", size=10)

    plt.rcParams["font.size"] = 10
    i = 0
    while i < vocab_size:
        #点プロット
        plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x")
    
        #文字プロット
        #plt.annotate(id_to_word[i+1], (data_pca[i][0], data_pca[i][1]), size=12)
    
        i += 1

    plt.savefig("./hoge_2d.pdf")
    plt.show()