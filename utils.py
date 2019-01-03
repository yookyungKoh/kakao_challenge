import torch
from torch.utils.data import Dataset, DataLoader
import os
import shutil

import fire
import h5py
import pickle
import numpy as np

class KakaoDataset(Dataset):
    def __init__(self, data_root, div, chunk_size=20000):
        data_path = os.path.join(data_root, 'data.h5py')
        self.data_root = data_root
        self.div = div
        self.data = self._load_data(data_path, div)
        self.chunk_size = chunk_size
        self.begin_offset = 0
        self.end_offset = self.begin_offset + chunk_size
        self.data_ptr = self._read_data(self.data)
        self.t_chunk = self.data_ptr[0][self.begin_offset:self.end_offset]
        self.f_chunk = self.data_ptr[1][self.begin_offset:self.end_offset]
        self.b_chunk = self.data_ptr[2][self.begin_offset:self.end_offset]
        self.m_chunk = self.data_ptr[3][self.begin_offset:self.end_offset]
        self.s_chunk = self.data_ptr[4][self.begin_offset:self.end_offset]
        self.d_chunk = self.data_ptr[5][self.begin_offset:self.end_offset]
        self.i_chunk = self.data_ptr[6][self.begin_offset:self.end_offset]
        self.total = len(self.data_ptr[0])
        meta_path = os.path.join(data_root, 'meta')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.y_vocab = meta['y_vocab']

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if not self.is_range(idx):
            # load next chunk.
            self.begin_offset = self.end_offset
            self.end_offset = min(self.begin_offset + self.chunk_size, self.total)
            self.t_chunk = self.data_ptr[0][self.begin_offset:self.end_offset]
            self.f_chunk = self.data_ptr[1][self.begin_offset:self.end_offset]
            self.b_chunk = self.data_ptr[2][self.begin_offset:self.end_offset]
            self.m_chunk = self.data_ptr[3][self.begin_offset:self.end_offset]
            self.s_chunk = self.data_ptr[4][self.begin_offset:self.end_offset]
            self.d_chunk = self.data_ptr[5][self.begin_offset:self.end_offset]
            self.i_chunk = self.data_ptr[6][self.begin_offset:self.end_offset]
        idx = idx - self.begin_offset
        X = list((self.t_chunk[idx], self.f_chunk[idx], self.i_chunk[idx]))
        y_cate = '{}>{}>{}>{}'.format(*self.global_onehot(self.b_chunk[idx], self.m_chunk[idx], self.s_chunk[idx], self.d_chunk[idx]))
        y = self.y_vocab[y_cate]
        if idx + self.begin_offset + 1 >= self.total:
            self.__init__(self.data_root, self.div, self.chunk_size)

        return X, y

    def global_onehot(self, b,m,s,d):
        b_i = np.argmax(b)+1
        m_i = np.argmax(m)+1
        s_i = np.argmax(s)+1
        d_i = np.argmax(d)+1
        if s_i == 1:
            s_i = -1
        if d_i == 1:
            d_i = -1
        return b_i, m_i, s_i, d_i

    def is_range(self, i):
        if self.begin_offset is not None and i < self.begin_offset:
            assert False, '%s < %s, index can not be lower than begin offset.' % (i, self.begin_offset)
        if self.end_offset is not None and self.end_offset <= i:
            return False
        return True

    def _load_data(self, data_path, div):
        data = h5py.File(data_path, 'r')
        return data[div]

    def _read_data(self, data):
#        pid = data['pid']
        text = data['uni']
        freq = data['w_uni']
        bcate = data['bcate']
        mcate = data['mcate']
        scate = data['scate']
        dcate = data['dcate']
        img = data['img_feat']

        return text, freq, bcate, mcate, scate, dcate, img

def split_h5py(path, out_path, div='dev', train_size=1400000, dev_size=200000):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    data_path = os.path.join(path, 'data.h5py')
    meta_path = os.path.join(path, 'meta')
    shutil.copy(meta_path, out_path)
    data = h5py.File(data_path, 'r')[div]
    out_path = os.path.join(out_path, 'data.h5py')
    out_data = h5py.File(out_path)
    train = out_data.create_group('train')
    dev = out_data.create_group('dev')
    for k in list(data.keys()):
        train[k] = data[k][:train_size]
        dev[k] = data[k][train_size:train_size+dev_size]
        
        

if __name__ == '__main__':
    fire.Fire({'split': split_h5py})
