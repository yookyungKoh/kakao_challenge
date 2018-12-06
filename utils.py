import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py

class KakaoDataset(Dataset):
    def __init__(self, data_root, chunk_size=20000):
        data_path = os.path.join(data_root, 'data.h5py')
        self.data_root = data_root
        self.data = self._load_data(data_path)
        self.chunk_size = chunk_size
        self.begin_offset = 0
        self.end_offset = self.begin_offset + chunk_size
        self.data_ptr = self._read_data(self.data)
#        self.p = chunk[0]
        self.t_chunk = self.data_ptr[0][self.begin_offset:self.end_offset]
        self.f_chunk = self.data_ptr[1][self.begin_offset:self.end_offset]
        self.b_chunk = self.data_ptr[2][self.begin_offset:self.end_offset]
        self.m_chunk = self.data_ptr[3][self.begin_offset:self.end_offset]
        self.s_chunk = self.data_ptr[4][self.begin_offset:self.end_offset]
        self.d_chunk = self.data_ptr[5][self.begin_offset:self.end_offset]
        self.total = len(self.data_ptr[0])

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
        idx = idx - self.begin_offset
        X = list((self.t_chunk[idx], self.f_chunk[idx]))
        y = list((self.b_chunk[idx], self.m_chunk[idx], self.s_chunk[idx], self.d_chunk[idx]))

        if idx + self.begin_offset + 1 >= self.total:
            self.__init__(self.data_root, self.chunk_size)

        return X, y

    def is_range(self, i):
        if self.begin_offset is not None and i < self.begin_offset:
            assert False, '%s < %s, index can not be lower than begin offset.' % (i, self.begin_offset)
        if self.end_offset is not None and self.end_offset <= i:
            return False
        return True

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


