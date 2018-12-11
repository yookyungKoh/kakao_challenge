import torch
import torch.nn as nn
import torch.nn.functional as F

from misc import get_logger, Option

opt = Option('./config.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, opt):
        super(MLP, self).__init__()
        vocab_size = opt.unigram_hash_size + 1
        max_len = opt.max_len
        self.cate_emb_size = 10
        
        self.cate1_size = 57
        self.cate2_size = 552
        self.cate3_size = 3190
        self.cate4_size = 404

        self.embd = nn.Embedding(vocab_size, opt.embd_size)
        self.b_emb = nn.Embedding(self.cate1_size, self.cate_emb_size)
        self.m_emb = nn.Embedding(self.cate2_size, self.cate_emb_size)
        self.s_emb = nn.Embedding(self.cate3_size, self.cate_emb_size)
        self.d_emb = nn.Embedding(self.cate4_size, self.cate_emb_size)

        self.linear1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(opt.embd_size, self.cate1_size))
        self.linear2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(opt.embd_size + self.cate_emb_size, self.cate2_size))
        self.linear3 = nn.Sequential(
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(opt.embd_size + 2*self.cate_emb_size, self.cate3_size))
        self.linear4 = nn.Sequential(
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(opt.embd_size + 3*self.cate_emb_size, self.cate4_size))

        self.init_weights()

    def init_weights(self):
        self.linear1[2].weight.data.uniform_(-0.05, 0.05)
        self.linear2[2].weight.data.uniform_(-0.05, 0.05)
        self.linear3[2].weight.data.uniform_(-0.05, 0.05)
        self.linear4[2].weight.data.uniform_(-0.05, 0.05)
        
    def forward(self, inputs):
        # inputs: (words, frequency)
        word_idx = inputs[0].type(torch.LongTensor).to(device) #(N, max_len)
        freq = inputs[1].type(torch.FloatTensor).to(device) #(N, max_len)

        word_embed = self.embd(word_idx) #(N, max_len, emb_size)
        text_feature = torch.bmm(word_embed.permute(0,2,1), freq.unsqueeze(2)) #(N, 128, 1)
        h1 = text_feature.squeeze() #(N, 128)
        
        out1 = self.linear1(h1) #(N, 57)
        y1 = torch.max(out1, dim=1)[1] #(N)
        y1 = self.b_emb(y1) #(N, 10)
    
        m_all = self.m_emb.weight
#        m_all = m_all.expand(y1.size(0), self.cate2_size, self.cate_emb_size) #(N, 552, 10)
        sim_bm = torch.matmul(m_all, y1.unsqueeze(2))
#        sim_bm = torch.bmm(m_all, y1.unsqueeze(2)) #(N, 552, 1)
        sim_bm = sim_bm.squeeze() #(N, 552)

        h2 = torch.cat((h1, y1), dim=1) #(N, 138)
        out2 = self.linear2(h2) #(N, 552)
        out2 = torch.mul(sim_bm, out2)
        y2 = torch.max(out2, dim=1)[1]
        y2 = self.m_emb(y2)

        s_all = self.s_emb.weight
#        s_all = s_all.expand(y2.size(0), self.cate3_size, self.cate_emb_size) #(N, 3190, 10)
        sim_ms = torch.matmul(s_all, y2.unsqueeze(2))
#        sim_ms = torch.bmm(s_all, y2.unsqueeze(2))
        sim_ms = sim_ms.squeeze()

        h3 = torch.cat((h2, y2), dim=1) #(N, 148)
        out3 = self.linear3(h3) #(N, 3190)
        out3 = torch.mul(sim_ms, out3)
        y3 = torch.max(out3, dim=1)[1]
        y3 = self.s_emb(y3)

        d_all = self.d_emb.weight
#        d_all = d_all.expand(y3.size(0), self.cate4_size, self.cate_emb_size) #(N, 404, 10)
        sim_sd = torch.matmul(d_all, y3.unsqueeze(2))
#        sim_sd = torch.bmm(d_all, y3.unsqueeze(2))
        sim_sd = sim_sd.squeeze()

        h4 = torch.cat((h3, y3), dim=1) #(N, 158)
        out4 = self.linear4(h4) #(N, 404)
        out4 = torch.mul(sim_sd, out4)
        
        return out1, out2, out3, out4

        
