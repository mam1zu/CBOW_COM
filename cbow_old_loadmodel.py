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
from utils import preprocess
from utils import create_contexts_target
from utils import cos_similarity
from WikiLoader import CustomDataset

args = sys.argv

class CBOWNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, weights=None, padding_idx=0):
        super(CBOWNet, self).__init__()
        self.weights = weights
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.input_emb = nn.Embedding(vocab_size, emb_dim)#パディングインデックスを追加するためvocab_sizeは実際の語彙数より+1となる
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        #Attention Word Embeddingでは、input_embとoutput_embは一致させるが、今のところは別とする

        #ネガティブサンプリングのやつ
        if self.weights is not None:
            wf = np.power(self.weights, 0.75)
            wf /= np.sum(wf)
            self.weights = torch.FloatTensor(wf)
        
    def forward(self, context_ids, centre_id):
        #コンテキストベクトルをあつめて平均を取り、output_embへの入力を得る
        #ここでアテンション重みをかけてあげればACBOWとなる...はず

        context_embedding = self.input_emb(context_ids)
        
        context_embedding = torch.mean(context_embedding, dim=1) #dim0: バッチ方向 dim1: 何？
        #次に、W_out, すなわちoutput_embの重みと内積を取る

        output = torch.matmul(context_embedding, self.output_emb.weight.T)

        #t = F.one_hot(torch.tensor(centre_id), num_classes=self.vocab_size)
        #t = t.view(1, self.vocab_size)

        #loss = F.cross_entropy(output, t.float())
        loss = F.cross_entropy(output, centre_id)
        #print("loss: " + str(loss))
        return loss
    
emb_dim = 300
model = CBOWNet(100001, emb_dim=emb_dim)
model.load_state_dict(torch.load("/tf/paper/cbow-com/models/dim_300/model_negs_7.pth"))

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


    pred_played = model.input_emb.weight[282] + said_minus_say
    ans_played = model.input_emb.weight[163]
    print(cos_similarity(pred_played, ans_played))
    # pred_went = model.input_emb.weight[582] + said_minus_say
    # ans_went = model.input_emb.weight[433]
    # print("cos similarity of pred and ans"); print(cos_similarity(pred_went, ans_went)) 

    print("Vector to make verb past :"); print(said_minus_say)
    #tokyo - japan + germany = ? (ans. berlin)
    while True:
        print("input first word id:")
        id_1 = int(input())

        if id_1 == "2231081":
            print("BYE")
            break
        
        print("input second word id:")
        id_2 = int(input())

        print(cos_similarity(model.input_emb.weight[id_1], model.input_emb.weight[id_2]))
