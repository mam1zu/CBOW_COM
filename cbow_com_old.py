# Coded by Kouki TAKASHIMA, Chiba Institute of Technology

import torch
import torch.nn as nn
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

        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)

        self.co_matrix = generate_co_occurrence_matrix(corpus, vocab_size, window_size=1)


        if self.weights is not None:
            wf = np.power(self.weights, 0.75)
            wf /= np.sum(wf)

            self.weights = torch.FloatTensor(wf)

    def forward(self, context_words, centre_word):

        context_embedding = self.input_emb(torch.tensor(context_words))

        comatrix_weight = []
        for idx in context_words:
            comatrix_weight.append(glove_weight_function(self.co_matrix[centre_word][idx]))
        comatrix_weight = torch.tensor(comatrix_weight)
        comatrix_weight_avg = torch.mean(comatrix_weight)
        comatrix_weight = comatrix_weight.view(-1, 1)
        context_embedding = torch.matmul(comatrix_weight.T, context_embedding)
        context_embedding = torch.divide(context_embedding, comatrix_weight_avg)

        output = torch.matmul(context_embedding, self.output_emb.weight.T)

        output = output.view(1, -1).squeeze()

        t = F.one_hot(torch.tensor(centre_word), num_classes=self.vocab_size)
        loss = F.cross_entropy(output, t.float())
        return loss
    
#ここから実行
sentence = "Japan is an island country in East Asia. Located in the Pacific Ocean off the northeast coast of the Asian mainland, it is bordered to the west by the Sea of Japan and extends from the Sea of Okhotsk in the north to the East China Sea in the south. Divided into 47 administrative prefectures and eight traditional regions, about 75% of the country's terrain is mountainous and heavily forested, concentrating its agriculture and highly urbanized population along its eastern coastal plains. With a population of over 123 million as of 2025, it is the 11th most populous country. The country's capital and largest city is Tokyo. "

corpus, word_to_id, id_to_word = preprocess(sentence)

model = CBOWNet(len(word_to_id), emb_dim=5, corpus=corpus)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

contexts, target = create_contexts_target(corpus, window_size=1)
batch_size = corpus.shape[0]

loss = model(contexts[0], target[0])

model.train()
for epoch in range(20):
    loss = 0
    model.zero_grad()

    for i in range(len(contexts)):
        loss += model(contexts[i], target[i])
    
    loss.backward()
    print("epoch :" + str(epoch))
    print("whole loss: " + str(loss))
    print("----------------------------")

    optimizer.step()

model.eval()
with torch.no_grad():
    print(model(contexts[0], target[0]))
    print(cos_similarity(model.input_emb.weight[word_to_id["japan"]], model.input_emb.weight[word_to_id["tokyo"]]))
    print(cos_similarity(model.output_emb.weight[word_to_id["japan"]], model.output_emb.weight[word_to_id["tokyo"]]))

    print(model.input_emb.weight)
    print(model.output_emb.weight)

