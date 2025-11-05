# CBOW normal

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
import gc
from functools import partial
from tqdm import tqdm as std_tqdm
from utils import cos_similarity, load_dict, glove_weight_function
from CustomLoader import CustomDataset#, WikiDataset
from scipy import sparse

device = 'cuda' if torch.cuda.is_available else 'cpu'
tqdm = partial(std_tqdm, dynamic_ncols=True)
window_size = 5
#動作確認用
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)

# custom_collate_fn
# 学習データに中心単語の共起情報を付与する
def collate_fn_dense(data_list):

    batch = default_collate(data_list)
    centre_id = batch['labels']
    context_ids = batch['data']
    #batch_size = centre_ids.shape[0]

    comat_row_idx = centre_id[:, torch.newaxis]
    comat_data = comat[comat_row_idx - 1, context_ids - 1].float()
    batch['comat_data'] = comat_data
    return batch


def suggested_collate_fn(data_list):
    
    batch = default_collate(data_list)
    
    centre_ids: np.ndarray = batch['labels'].numpy()
    context_ids: np.ndarray = batch['data'].numpy()
    batch_size: int = centre_ids.shape[0]
    rows_comat_csr = comat[centre_ids]
    rows_comat_dense = rows_comat_csr.toarray()
    row_idx_gather = np.arange(batch_size).reshape(-1, 1)
    batch_comat_data = rows_comat_dense[row_idx_gather, context_ids]

    assert batch_comat_data.shape == (batch_size, window_size * 2)
    
    batch['comat_data'] = torch.from_numpy(batch_comat_data)
    return batch

def collate_fn_fancyindexing(data_list):
    batch : list = default_collate(data_list)
    centre_ids  : np.ndarray = batch['labels'].numpy()        # shape: [batch_size]
    context_ids : np.ndarray = batch['data'].numpy()         # shape: [batch_size, window_size*2]
    batch_size = centre_ids.shape[0]
    hoge = np.zeros((batch_size, window_size * 2), dtype=np.uint8)
    rows_comat = comat[centre_ids]

    row_indices = np.repeat(np.arange(batch_size), window_size * 2)
    col_indices = context_ids.flatten()
    
    batch['comat_data'] = torch.from_numpy(rows_comat[row_indices, col_indices].A.flatten().reshape(batch_size, window_size*2))

    # for i in range(batch_size):
    #     row_comat = rows_comat.getrow(i)
    #     row_comat = row_comat[:, context_ids[i]].todense()
    #     hoge[i] = row_comat
    
    # batch['comat_data'] = torch.from_numpy(hoge)
    return batch

def custom_collate_fn(data_list):
    
    batch       : list       = default_collate(data_list)
    batch['data'], _ = torch.sort(batch['data'])
    centre_ids  : np.ndarray = batch['labels'].numpy()        # shape: [batch_size]
    context_ids : np.ndarray = batch['data'].numpy()         # shape: [batch_size, window_size*2]
    batch_size  : int        = centre_ids.shape[0]
    
    rows_comat  : sparse.csr_matrix = comat[centre_ids]
    #assert rows_comat.shape == (batch_size, vocab_size+1)
    #del centre_ids

    mask_col    : np.ndarray        = np.int32(context_ids.flatten())
    #assert mask_col.size == batch_size*window_size*2
    #del context_ids

    mask_row    : np.ndarray        = np.int32(np.repeat(np.arange(batch_size), window_size * 2))
    #assert mask_row.size == batch_size*window_size*2
    mask = np.ndarray(shape=(batch_size, vocab_size), dtype=np.uint8)
    mask[mask_row, mask_col] = 1

    data        : np.ndarray        = np.ones(mask_col.shape[0], dtype=np.int8)
    #assert data.size == batch_size*window_size*2

    #rows_mask   : sparse.coo_matrix = sparse.coo_matrix((data, (mask_row, mask_col)), shape=(batch_size, vocab_size+1))
    #rows_mask   : sparse.csr_matrix = rows_mask.tocsr() # shape: [batch_size, vocab_size], size: [batch_size*window_size*2]
    """
    一つの学習データに同一の周辺単語が複数回発生するとき
    マスク用の行列をCSR行列に変換する際, ある中心単語とその周辺単語の要素'1'が合算され2以上になる
    このとき非ゼロ要素の個数はbatch_size * window_size * 2より小さくなり,
    アダマール積後の疎行列のdata配列を用いて非ゼロ要素のみ取り出し,
    context_idsのshapeと一致させるという手法は正常に動作しない
    """

    #assert rows_mask.shape == (batch_size, vocab_size+1)
    #comat_data  : sparse.csr_matrix = rows_comat.multiply(rows_mask)
    comat_data = rows_comat.multiply(mask)
    #assert comat_data.shape == (batch_size, vocab_size+1)
    #print(f"comat_data.data.size: {comat_data.data.size}, expected: {batch_size*window_size*2}")
    #assert comat_data.data.size == batch_size*window_size*2
    #comat_data_kari = np.ones(shape=(batch_size, window_size*2), dtype=np.uint8)
    #せめてアダマール積が重いかどうかだけ判断
    batch['comat_data'] = torch.from_numpy(comat_data_kari)
    #batch['comat_data'] = torch.from_numpy(np.reshape(comat_data.data, [batch_size, window_size*2]))
    del rows_comat, rows_mask, comat_data
    return batch


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
    

print("CBOW-COM")
num_negative_samples = 5
emb_dim = 300
vocab_size = 100000
overall_start_time = time.time()
print("Loading cooccurrence matrix from disk to RAM...")
#comat: sparse.csr_matrix = sparse.load_npz("./comat.window_5.vocab_400000.uint8.noweight.npz")
comat: np.ndarray = np.load("./comat_dense.window_5.vocab_100000.float16.noweight.npy")
comat = torch.from_numpy(comat)
print("Transferring to VRAM...")
print("skip")
#comat = torch.tensor(comat).to(device)
#comat = torch.tensor(comat)
print("Loaded to VRAM")

model = CBOWNet(vocab_size+1, emb_dim=emb_dim,)
model.load_state_dict(torch.load(f"./models/dim_{emb_dim}_{vocab_size}/model_CBOWCOM_log_AdamW_default_negs_5_epoch_2_117.pth"))
model = model.to(device)
optimizer = optim.AdamW(model.parameters())
weights = torch.from_numpy(np.load("./wiki-cleaned.nostopword.100000.negdist.npy")).to(device, dtype=torch.float) #negative sampling weights

word_to_id, id_to_word = load_dict("/tf/paper/cbow-com/utils/wiki-cleaned.nostopword.100000.vocab")
if word_to_id is None:
    print("Vocabulary file can't be opened, Abort!")
    sys.exit(1)

exponential_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#optimizer = optim.SGD(model.parameters(), lr=0.025)

# with tqdm(range(epoch)) as pbar_epoch:
#     for e in pbar_epoch:
#         loss = 0
#         pbar_epoch.set_description("[Epoch %d]" % (e))
#         with tqdm(enumerate(data_loader), total=len(data_loader)) as pbar_loss:
#             for i, (data, labels) in pbar_loss:
#                 data, labels = data.to(device), labels.to(device)
#                 loss += model(data, labels)
#                 loss.backward()
#                 optimizer.step()

for epoch in tqdm(range(3, 5)):
    epoch_start_time = time.time()
    dataset_idx = 0
    if dataset_idx == -1:
        break
    while True:
        loss_all = 0
        dataset = np.load(f"/tf/paper/cbow-com/dataset/window_5/wiki-cleaned.nostopword.100000.train.{dataset_idx:04d}")
        data = dataset[:, 1:] #context
        labels = dataset[:, 0] #centre
        custom_dataset= CustomDataset(data, labels)
        #custom_dataset.data = torch.from_numpy(custom_dataset.data).to(device)
        #custom_dataset.labels = torch.from_numpy(custom_dataset.labels).to(device)
        data_loader = DataLoader(
            dataset=custom_dataset,
            batch_size=32768,
            shuffle=True,
            num_workers=12,
            collate_fn=collate_fn_dense,
            pin_memory=True,
            )
        for batch in tqdm(data_loader):
            context_ids = batch['data'].to(device, non_blocking=True, dtype=torch.long)
            centre_ids = batch['labels'].to(device, non_blocking=True, dtype=torch.long)
            comat_data = batch['comat_data'].to(device, non_blocking=True, dtype=torch.float32)
            #共起行列から行を取り出しGPUに転送する処理
            # centre_ids_numpy = centre_ids.numpy()
            # comat_data = comat[centre_ids_numpy]
            # comat_data = comat_data.tocoo()
            # indices = torch.from_numpy(np.vstack((comat_data.row, comat_data.col))).long()
            # values = torch.from_numpy(comat_data.data)
            # shape = torch.Size(comat_data.shape)
            # comat_data = torch.sparse_coo_tensor(indices, values, shape).to(device, non_blocking=True)
            #終了

            #centre_idsは共起行列作成のためにメモリに保留していたが処理が完了したのでGPUに転送する
            neg_size = centre_ids.shape[0]
            neg_ids = torch.multinomial(weights, neg_size * num_negative_samples, replacement=True)
            neg_ids = neg_ids.view(neg_size, num_negative_samples) # shape: (batch_size, num_negative_samples)

            optimizer.zero_grad()
            loss = model(context_ids, centre_ids, neg_ids, comat_data)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        
        print(f"loss: {loss_all}")
        #if dataset_idx % 25 == 0:
        #    torch.save(model.state_dict(), f"./models/dim_{emb_dim}_{vocab_size}/model_CBOWCOM_log_AdamW_default_negs_{num_negative_samples}_epoch_{epoch}_{dataset_idx}.pth")
        if dataset_idx == 117:
            torch.save(model.state_dict(), f"./models/dim_{emb_dim}_{vocab_size}/model_CBOWCOM_log_AdamW_default_negs_{num_negative_samples}_epoch_{epoch}_{dataset_idx}.pth")
            break
        dataset_idx += 1
    epoch_end_time = time.time()
    print(f"epoch {epoch}: {epoch_end_time - epoch_start_time}[s]")

overall_end_time = time.time()
print(f"overall duration: {overall_end_time - overall_start_time}[s]")

    
# for epoch in range(5):
#     loss = 0
#     model.train()
#     optimizer.zero_grad()


#     for batch in data_loader:
#         loss += model(batch['data'], batch['labels'])

#     for i in range(len(contexts)):
#         #一気にコーパスをロードするとバッファオーバーフローを引き起こす可能性がある
#         #そこで、一度にロードするコーパス量を制限する。単語IDがint32であるとすると、1億トークンであれば学習用のデータをコンテキストワードとセンターワードで
#         #4.4GB程度になると概算. パイプライン化などしてオーバーヘッドを減らせるか？

#         loss += model(contexts[i], target[i])
    
#     loss.backward()
#     print("epoch :" + str(epoch))
#     print("whole_loss: " + str(loss))
#     print("--------")
#     optimizer.step()

model.eval()
with torch.no_grad():
    #tokyo - japan + germany = ? (ans. berlin)
    pred = model.input_emb.weight[1769] - model.input_emb.weight[463] + model.input_emb.weight[1213]
    ans = model.input_emb.weight[479]

    print("predicted berlin :"); print(pred)
    print("True      berlin :"); print(ans)
    print("cos similarity: "); print(cos_similarity(pred, ans))
    print("announcement , news"); print(cos_similarity(model.input_emb.weight[5178], model.input_emb.weight[583]))
    print("cup, coffee"); print(cos_similarity(model.input_emb.weight[193], model.input_emb.weight[4395]))
    print("japan, japanese"); print(cos_similarity(model.input_emb.weight[463], model.input_emb.weight[542]))
    print(model.input_emb.weight)
    print(model.output_emb.weight)
