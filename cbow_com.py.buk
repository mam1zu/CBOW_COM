# CBOW with Co-Occurrence Matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from utils import preprocess, create_contexts_target, cos_similarity, generate_co_occurrence_matrix, glove_weight_function

device = 'cuda' if torch.cuda.is_available else 'cpu'

class CBOWNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, corpus, weights=None, padding_idx=0):
        super(CBOWNet, self).__init__()
        self.weights = weights
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.input_emb = nn.Embedding(vocab_size, emb_dim)#パディングインデックスを追加するためvocab_sizeは実際の語彙数より+1となる
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self.co_matrix = generate_co_occurrence_matrix(corpus, vocab_size, window_size=1)
        #Attention Word Embeddingでは、input_embとoutput_embは一致させるが、今のところは別とする

        #ネガティブサンプリングのやつ
        if self.weights is not None:
            wf = np.power(self.weights, 0.75)
            wf /= np.sum(wf)
            self.weights = torch.FloatTensor(wf)
        
    def forward(self, context_words, centre_word):
        #コンテキストベクトルをあつめて平均を取り、output_embへの入力を得る
        #ここでアテンション重みをかけてあげればACBOWとなる...はず

        context_embedding = self.input_emb(torch.tensor(context_words))

        comatrix_weight = []
        for idx in context_words:
            comatrix_weight.append(glove_weight_function(self.co_matrix[centre_word][idx]))
        comatrix_weight = torch.tensor(comatrix_weight)
        comatrix_weight_avg = torch.mean(comatrix_weight)
        comatrix_weight = comatrix_weight.view(-1, 1)
        context_embedding = torch.matmul(comatrix_weight.T, context_embedding)
        context_embedding = torch.divide(context_embedding, comatrix_weight_avg)

        
        #context_embedding = torch.mean(context_embedding, dim=0)
        #次に、W_out, すなわちoutput_embの重みと内積を取る

        output = torch.matmul(context_embedding, self.output_emb.weight.T)
        output = output.view(1, -1).squeeze()

        t = F.one_hot(torch.tensor(centre_word), num_classes=self.vocab_size)
        #t = t.view(1, self.vocab_size)
        loss = F.cross_entropy(output, t.float())
        #print("context_embedding :" + str(context_embedding))
        #print("output : " + str(F.softmax(output)))
        #print("t_vec  : " + str(t))
        #print("loss: " + str(loss))
        return loss
    

#sentence = "You say goodbye and I say hello."
sentence = "Japan is an island country in East Asia. Located in the Pacific Ocean oof the northeast coast of the Asian mainland, it is borderd on the west by the Sea of Japan and extends from the Sea of Okhotsk in the north to the East China Sea in the south. The Japanese achipelago consists of four major islands, Hokkaido, Honshu, Shikoku, and Kyushu, and thousands of smaller islands, covering 377,975 square kilometers. Japan has a population of over 123 million as of 2025, making it the eleventh-most populous country. The capital of Japan and its largest city is Tokyo. the Greater Tokyo Area is the largest metropolitan area in the world."
num_negative_samples = 3
corpus, word_to_id, id_to_word = preprocess(sentence)

model = CBOWNet(len(word_to_id), emb_dim=5, corpus=corpus)
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.SGD(model.parameters(), lr=0.025)
contexts, target = create_contexts_target(corpus, window_size=1)
batch_size = corpus.shape[0]

loss = model(contexts[0], target[0])

for epoch in range(20):
    loss = 0
    model.train()
    optimizer.zero_grad()
    for i in range(len(contexts)):
        loss += model(contexts[i], target[i])
    
    loss.backward()
    print("epoch :" + str(epoch))
    print("whole_loss: " + str(loss))
    print("--------")
    optimizer.step()

model.eval()
with torch.no_grad():
    print(model(contexts[0], target[0]))
    print(cos_similarity(model.input_emb.weight[word_to_id["japan"]], model.input_emb.weight[word_to_id["tokyo"]]))
    print(cos_similarity(model.output_emb.weight[word_to_id["japan"]], model.output_emb.weight[word_to_id["tokyo"]]))

    print(model.input_emb.weight)
    print(model.output_emb.weight)