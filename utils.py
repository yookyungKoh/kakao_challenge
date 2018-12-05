import torch
from torch.utils.data import Dataset
import os

class KakaoDataset(Dataset):
    def __init__(self, data_root):
        data_path = os.path.join(data_root, 'data.h5py')
        self.data = self._load_data(data_path)
        self.X, self.y = self._read_data(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]
    
    def _load_data(self, data_path):
        return h5py.File(data_path)

    def _read_data(self, data):
        pid = data['pid']
        text = data['uni']
        freq = data['w_uni']
        bcate = data['bcate']
        mcate = data['mcate']
        scate = data['scate']
        dcate = data['dcate']
        
        X = (pid, text, freq)
        y = (bcate, mcate, scate, dcate)

        return X, y
