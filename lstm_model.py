import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

from misc import get_logger, Option

opt = Option('./config.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextOnlyLSTM(nn.Module):
    def __init__(self, opt):
        super(TextOnlyLSTM, self).__init__()
        vocab_size = opt.unigram_hash_size + 1
        max_len = opt.max_len
        self.cate_emb_size = 128
        self.hidden_size = 128

        cate1_size = 57
        cate2_size = 552
        cate3_size = 3190
        cate4_size = 404

        self.embd = nn.Embedding(vocab_size, opt.embd_size)

        self.bcate_embd = nn.Embedding(cate1_size, self.cate_emb_size)
        self.mcate_embd = nn.Embedding(cate2_size, self.cate_emb_size)
        self.scate_embd = nn.Embedding(cate3_size, self.cate_emb_size)
        self.dcate_embd = nn.Embedding(cate4_size, self.cate_emb_size)

        self.lstm = nn.LSTM(opt.embd_size, self.hidden_size, batch_first=True)

        self.bcate_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, cate1_size))
        self.mcate_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, cate2_size)
        )
        self.scate_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, cate3_size))
        self.dcate_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, cate4_size)
        )

    def forward(self, inputs, cates, val=False):
        # inputs: (words, frequency)
        word_idx = inputs[0].type(torch.LongTensor).to(device)  # (N, max_len)
        freq = inputs[1].type(torch.FloatTensor).to(device)  # (N, max_len)
        word_embed = self.embd(word_idx)  # (N, max_len, emb_size)
        text_feature = torch.bmm(word_embed.permute(0, 2, 1),
                                 freq.unsqueeze(2))  # (N, emb_size, max_len) b* (N, max_len, 1) -> (N, 128,1)
        h1 = text_feature.squeeze()  # (N, 128)

        if val:
            h1, cell = self.lstm(h1.unsqueeze(1))
            out1 = self.bcate_linear(h1).squeeze()  # (N, 57)
            bcate_embd = self.bcate_embd(out1.argmax(dim=1))

            h2, cell = self.lstm(bcate_embd.unsqueeze(1), cell)
            out2 = self.mcate_linear(h2).squeeze()
            mcate_embd = self.mcate_embd(out2.argmax(dim=1))

            h3, cell = self.lstm(mcate_embd.unsqueeze(1), cell)
            out3 = self.scate_linear(h3).squeeze()
            scate_embd = self.scate_embd(out3.argmax(dim=1))

            h4, cell = self.lstm(scate_embd.unsqueeze(1), cell)
            out4 = self.dcate_linear(h4).squeeze()

        else:
            bcate, mcate, scate, _ = [cate.argmax(dim=1).long() for cate in cates]  # (N,1), (N,1), (N,1), _ = (N,4)
            bcate_embd = self.bcate_embd(bcate)  # (N, 1, emb_size)
            mcate_embd = self.mcate_embd(mcate)
            scate_embd = self.scate_embd(scate)
            embeddings = torch.cat(
                (h1.unsqueeze(1), bcate_embd.unsqueeze(1), mcate_embd.unsqueeze(1), scate_embd.unsqueeze(1)),
                1)  # (N, 4, emb_size=128)
            hiddens, _ = self.lstm(embeddings)
            h1, h2, h3, h4 = [hiddens[:, time_step].squeeze() for time_step in range(4)]
            out1 = self.bcate_linear(h1)  # (N, 57)
            out2 = self.mcate_linear(h2)  # (N, 552)
            out3 = self.scate_linear(h3)  # (N, 3190)
            out4 = self.dcate_linear(h4)  # (N, 404)

        out = torch.cat((out1, out2, out3, out4), dim=1)  # (N, 4203)

        return out1, out2, out3, out4


def make_linear(in_dim, out_dim):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_dim, out_dim)
    )


class MLP(nn.Module):
    def __init__(self, opt, meta_path):
        super(MLP, self).__init__()
        vocab_size = opt.unigram_hash_size + 1
        max_len = opt.max_len
        image_feature_size = 2048
        text_feature_size = opt.embd_size
        self.embd = nn.Embedding(vocab_size, opt.embd_size)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.num_classes = len(meta['y_vocab'])
        self.linear = make_linear(image_feature_size + text_feature_size, self.num_classes)

    def forward(self, inputs):
        word_idx = inputs[0].type(torch.LongTensor).to(device)  # (N, max_len)
        freq = inputs[1].type(torch.FloatTensor).to(device)  # (N, max_len)
        image_feature = inputs[2].type(torch.FloatTensor).to(device)  # (N, 2048)

        word_embed = self.embd(word_idx)  # (N, max_len, emb_size)
        text_feature = torch.bmm(word_embed.permute(0, 2, 1), freq.unsqueeze(2))  # (N, 128, 1)
        h1 = text_feature.squeeze()  # (N, 128)
        input_feat = torch.cat((h1,image_feature), 1)  # (N, 128+2048)
        out = self.linear(input_feat)
        return out
