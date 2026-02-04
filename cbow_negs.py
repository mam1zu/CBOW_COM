# CBOW normal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import time
import glob
import random
from functools import partial
from tqdm import tqdm as std_tqdm
from utils import preprocess
from utils import create_contexts_target
from utils import cos_similarity, load_vocab, load_dict
from utils import _init_normal
from ws353benchmark import get_dataframe
from CustomLoader import CustomDataset
from WikiLoader import CBOWDataset
from scipy.stats import spearmanr, rankdata
device = 'cuda' if torch.cuda.is_available else 'cpu'
tqdm = partial(std_tqdm, dynamic_ncols=True)
random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

#default hyper-parameter settings
window_size = 5
num_negative_samples = 5
emb_dim = 300

args = sys.argv
print("CBOW Normal with negative sampling")
if len(args) <= 1:
    print("no parameters are set, use default settings")
elif len(args) != 4:
    print("num of parameters doesn't match, abort!")
    print(f"example: python3 {args[0]} [window_size] [num_negs]")
    sys.exit(1)
else:
    window_size = int(args[1])
    num_negative_samples = int(args[2])
    emb_dim = int(args[3])
print(f"window_size: {window_size}")
print(f"num_negative_samples: {num_negative_samples}")
print(f"emb_dim: {emb_dim}")

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
            cos_sim.append(F.cosine_similarity(word1_vector.to('cpu'), word2_vector.to('cpu'), dim=0).item())
            human_value_list.append(human_value)
        except KeyError:
            continue

    spearman_correlation, _ = spearmanr(cos_sim, human_value_list)
    return spearman_correlation

def _init_normalized():
    #Word2Vec original implementation
    padding_vector = np.zeros((1, emb_dim), dtype=np.float32)
    init_weights = np.random.uniform(
        size=(vocab_size, emb_dim),
        low  = -0.5 / emb_dim,
        high = 0.5 / emb_dim,
        ).astype(np.float32)
    init_weights = torch.from_numpy(np.concatenate((padding_vector, init_weights)))
    return init_weights

def _init_normal():
    padding_vector = np.zeros((1, emb_dim),dtype=np.float32)
    print(padding_vector)
    print(padding_vector.shape)
    init_weights = np.random.normal(
        size=(vocab_size, emb_dim),
        loc = 0.0,
        scale = 0.1
    ).astype(np.float32)
    print(init_weights)
    print(init_weights.shape)

    init_weights = torch.from_numpy(np.concatenate((padding_vector, init_weights)))

    return init_weights

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
model = CBOWNet(vocab_size+1, emb_dim=emb_dim)
word_to_id, id_to_word = load_dict('./utils/wiki-cleaned.nostopword.100000.vocab')

init_input_weights = _init_normalized()
init_output_weights = _init_normalized()
model.input_emb.weight.data = init_input_weights
model.output_emb.weight.data = init_output_weights
norms = torch.linalg.norm(model.input_emb.weight.data, dim=1)
print(norms.mean().item(), norms.std().item())

#model.input_emb.weight = nn.Parameter()
#model.load_state_dict(torch.load(f"./models/dim_{emb_dim}_{vocab_size}_hozon/model_AdamW_default_negs_5_epoch_2_117.pth"))
model = model.to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.05)
# scheduler = torch.optim.lr_scheduler.LinearLR(
#     optimizer,
#     start_factor=1.0,
#     end_factor=0.002,
#     total_iters= 5 * 60416
# )
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=5,
#     eta_min=1e-5
# )

weights = torch.from_numpy(np.load(f"./utils/wiki-cleaned.nostopword.{vocab_size}.negdist.npy")).to(device, dtype=torch.float) #negative sampling weights
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

vocab_list = load_vocab('./utils/wiki-cleaned.nostopword.100000.vocab')
# cbowdataset = CBOWDataset(
#     path='./utils/wiki-cleaned.nostopword.100000.txt',
#     num_lines=5772938,
#     num_line_per_chunk=50000,
#     window_size=5,
#     word_vocab=vocab_list
# )
# data_loader = DataLoader(
#     dataset=cbowdataset,
#     batch_size=1,
#     num_workers=24,
#     pin_memory=True,
#     shuffle=False,
#     collate_fn=cbowdataset.collate_fn
# )

print("CBOW - Normal with Negative Sampling")
# print(cbowdataset, data_loader)
# for epoch in tqdm(range(0, 5)):
#     iteration_idx = 0
#     for batch in tqdm(data_loader):
#         centre_ids = batch[0].to(device, non_blocking=True, dtype=torch.long)
#         context_ids = batch[1].to(device, non_blocking=True, dtype=torch.long)

#         batch_size = centre_ids.shape[0]
#         neg_ids = torch.multinomial(weights, batch_size * num_negative_samples, replacement=True)
#         neg_ids = neg_ids.view(batch_size, num_negative_samples)

#         optimizer.zero_grad()
#         loss = model(context_ids, centre_ids, neg_ids)
#         loss.backward()
#         optimizer.step()
#         #print(f"epoch: {epoch}, itr: {iteration_idx}, loss: {loss.item()}")
#     scheduler.step()
#     torch.save(model.state_dict(), f"./models/dim_{emb_dim}_{vocab_size}/model_AdamW_default_init_normalized_negs_{num_negative_samples}_epoch_{epoch}.pth")


dataset_path = glob.glob(f"/tf/paper/cbow-com/dataset/window_{window_size}/wiki-cleaned.*")
dataset_path.sort()
num_dataset_files = len(dataset_path)

for epoch in tqdm(range(0, 5)):
    start_time = time.time()
    dataset_idx = 0
    if dataset_idx == -1:
        break
    while dataset_idx+1 <= num_dataset_files:
        loss_all = 0
        dataset = np.load(dataset_path[dataset_idx])
        #dataset = np.load(f"/tf/paper/cbow-com/dataset/window_{window_size}/wiki-cleaned.nostopword.{vocab_size}.train.{dataset_idx:04d}")
        data = dataset[:, 1:] #context
        labels = dataset[:, 0] #centre
        custom_dataset= CustomDataset(data, labels)
        #custom_dataset.data = torch.from_numpy(custom_dataset.data).to(device)
        #custom_dataset.labels = torch.from_numpy(custom_dataset.labels).to(device)
        data_loader = DataLoader(
            dataset=custom_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=16, 
            pin_memory=True,
            )
        for batch in tqdm(data_loader):
            context_ids = batch['data'].to(device, non_blocking=True, dtype=torch.long)
            centre_ids = batch['labels'].to(device, non_blocking=True, dtype=torch.long)

            batch_size_negs = centre_ids.shape[0]
            neg_ids = torch.multinomial(weights, batch_size_negs * num_negative_samples, replacement=True)
            neg_ids = neg_ids.view(batch_size_negs, num_negative_samples) # shape: (batch_size, num_negative_samples)

            optimizer.zero_grad()
            loss = model(context_ids, centre_ids, neg_ids)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        norms = torch.linalg.norm(model.input_emb.weight.data, dim=1)
        print(f"dataset_idx: {dataset_idx} loss:{loss_all}, avg_norm: {norms.mean().item()}, std: {norms.std().item()}")
        print(f"WS353: {eval_benchmark('ws353')}, SimLex-999: {eval_benchmark('simlex999')}")
        # if dataset_idx % 25 == 0:
        #     torch.save(model.state_dict(), f"./models/dim_{emb_dim}_{vocab_size}/model_Adam_lr_3e-4_ExpLR_init_normalized_negs_{num_negative_samples}_epoch_{epoch}_{dataset_idx}.pth")
        dataset_idx += 1
    torch.save(model.state_dict(), f"./models/dim_{emb_dim}/window_{window_size}/model_CBOW_AdamW_default_batch_{batch_size}_init_normalized_negs_{num_negative_samples}_epoch_{epoch}.pth")
    end_time = time.time()
    with open("./experiment_duration_log.txt", 'a') as duration_log:
        duration_log.write(f"CBOW {window_size} {num_negative_samples} {emb_dim} {epoch}: {end_time - start_time}\n")

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
