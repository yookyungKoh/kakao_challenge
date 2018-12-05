import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py

class KakaoDataset(Dataset):
    def __init__(self, data_root):
        data_path = os.path.join(data_root, 'data.h5py')
        self.data = self._load_data(data_path)
        chunk = self._read_data(self.data)
#        self.p = chunk[0]
        self.t = chunk[0]
        self.f = chunk[1]
        self.b = chunk[2]
        self.m = chunk[3]
        self.s = chunk[4]
        self.d = chunk[5]
         
    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        X = list((self.t[idx], self.f[idx]))
        y = list((self.b[idx], self.m[idx], self.s[idx], self.d[idx]))
        return X, y

    def _load_data(self, data_path):
        data = h5py.File(data_path, 'r')
        if 'train' in data_path:
            return data['train']
        elif 'dev' in data_path:
            return data['dev']

    def _read_data(self, data):
#        pid = data['pid']
        text = data['uni']
        freq = data['w_uni']
        bcate = data['bcate']
        mcate = data['mcate']
        scate = data['scate']
        dcate = data['dcate']
         
        return text, freq, bcate, mcate, scate, dcate

        
