# CBOW normal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
from utils import preprocess
from utils import create_contexts_target
from utils import cos_similarity
from utils import _init_normal
from WikiLoader import CustomDataset
device = 'cuda' if torch.cuda.is_available else 'cpu'

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

        #ネガティブサンプリングのやつ
        if self.weights is not None:
            wf = np.power(self.weights, 0.75)
            wf /= np.sum(wf)
            self.weights = torch.FloatTensor(wf)
        
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
    

num_negative_samples = 10

emb_dim = 300
model = CBOWNet(100001, emb_dim=emb_dim)
#model.input_emb.weight = nn.Parameter()
#model.load_state_dict(torch.load("./model_2.pth"))
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
weights = torch.from_numpy(np.load("./wiki-cleaned.nostopword.100000.negdist.npy")).to(device, dtype=torch.float) #negative sampling weights
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

for epoch in tqdm(range(1)):
    dataset_idx = 0
    loss_all = 0
    if dataset_idx == -1:
        break

    while True:
        dataset = np.load(f"/tf/paper/cbow-com/dataset/window_10/wiki-cleaned.nostopword.100000.train.{dataset_idx:04d}")
        data = dataset[:, 1:] #context
        labels = dataset[:, 0] #centre
        custom_dataset= CustomDataset(data, labels)
        custom_dataset.data = torch.from_numpy(custom_dataset.data).to(device)
        custom_dataset.labels = torch.from_numpy(custom_dataset.labels).to(device)
        data_loader = DataLoader(dataset=custom_dataset, batch_size=8192, shuffle=True)
        for batch in tqdm(data_loader):
            context_ids = batch['data'].to(device, dtype=torch.long)
            centre_ids = batch['labels'].to(device, dtype=torch.long)

            batch_size = centre_ids.shape[0]
            neg_ids = torch.multinomial(weights, batch_size * num_negative_samples, replacement=True)
            neg_ids = neg_ids.view(batch_size, num_negative_samples) # shape: (batch_size, num_negative_samples)

            optimizer.zero_grad()
            loss = model(context_ids, centre_ids, neg_ids)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        torch.save(model.state_dict(), f"./models/dim_{emb_dim}/model_negs_{dataset_idx}.pth")
        if dataset_idx == 20:
            break
        dataset_idx += 1
    print("epoch {epoch} loss_all: {loss_all:.4f}")
    loss_all = 0
    
    
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
