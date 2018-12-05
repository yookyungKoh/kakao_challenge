import torch
import torch.nn as nn
import torch.nn.functional as F

from misc import get_logger, Option

opt = Option('./config.json')

class TextOnly(nn.Module):
    def __init__(self, opt):
        super(TextOnly, self).__init__()
        vocab_size = opt.unigram_hash_size + 1
        max_len = opt.max_len
        self.cate_emb_size = 10
        
        cate1_size = 57
        cate2_size = 552
        cate3_size = 3190
        cate4_size = 404

        self.embd = nn.Embedding(vocab_size, opt.embd_size)
        self.cate1_emb = nn.Embedding(cate1_size, self.cate_emb_size)
        self.cate2_emb = nn.Embedding(cate2_size, self.cate_emb_size)
        self.cate3_emb = nn.Embedding(cate3_size, self.cate_emb_size)
    
        self.linear1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(opt.embd_size, cate1_size))
        self.linear2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(opt.embd_size + self.cate_emb_size, cate2_size))
        self.linear3 = nn.Sequential(
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(opt.embd_size + 2*self.cate_emb_size, cate3_size))
        self.linear4 = nn.Sequential(
                nn.Dropout(0.5)),
                nn.ReLU(),
                nn.Linear(opt.embd_size + 3*self.cate_emb_size, cate4_size),

        self.init_weights()

    def init_weights(self):
        
    def forward(self, inputs):
        # inputs: (words, frequency)
        word_idx = inputs[/] #(max_len)
        freq = inputs[/] #(max_len)
        
        word_embed = self.embd(word_idx) #(N, max_len, emb_size)
        text_feature = torch.mm(word_embed.permute(1,0), freq.unsqueeze(1)) #(N, 128,1)
        h1 = text_feature.squeeze() #(N, 128)
        out1 = self.linear1(h1) #(N, 57)
        y1 = torch.max(out1, dim=1)[1] #(N)
        y1 = self.cate1_emb(y1) #(N, 10)
        h2 = torch.cat((h1, y1), dim=1) #(N, 138)
        out2 = self.linear2(h2) #(N, 552)
        y2 = torch.max(out2, dim=1)[1]
        y2 = self.cate2_emb(y2) 
        h3 = torch.cat((h2, y2), dim=1) #(N, 148)
        out3 = self.linear3(h3) #(N, 3190)
        y3 = torch.max(out3, dim=1)[1] 
        y3 = self.cate3_emb(y3)
        h4 = torch.cat((h3, y3), dim=1) #(N, 158)
        out4 = self.linear(h4) #(N, 404)
        
        out = torch.cat((out1, out2, out3, out4), dim=1) #(N, 4203)

        return out

        
