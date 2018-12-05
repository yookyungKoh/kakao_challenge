import torch
from torch.utils.data import Dataset, DataLoader
import os

class KakaoDataset(Dataset):
    def __init__(self, data_root):
        data_path = os.path.join(data_root, 'data.h5py')
        meta_path = os.path.join(data_root, 'meta')
        self.data = h5py.File(data_path, 'r')
        self.meta = cPickle.loads(open(meta_path, 'rb').read())
        self.y = self.meta['y_vocab']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]

