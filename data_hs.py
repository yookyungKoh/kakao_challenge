import h5py
import re
from multiprocessing.pool import Pool
import mmh3
import numpy as np
from collections import Counter

from misc import Option

opt = Option('./config.json')
re_sc = re.compile('[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')


def sanitize(product):
    product = re_sc.sub(' ', product).strip().split()
    words = [w.strip() for w in product]
    words = [w for w in words
             if opt.min_word_length <= len(w) < opt.max_word_length]
    return words


def decode(str_list, protocol='utf-8'):
    return [s.decode(protocol) for s in str_list]


def load_data_col(path_list, col, div='train', str_decode=False):
    result = []
    for path in path_list:
        data = h5py.File(path, 'r')
        data = data[div][col][:]
        if str_decode:
            data = decode(data)
        result.extend(data)
    return result


def hash_func_list(word_list):
    return [mmh3.hash(word) % opt.unigram_hash_size + 1 for word in word_list]


def to_numpy(word_freq):
    words_np = np.zeros(opt.max_len, dtype=np.float32)
    freqs_np = np.zeros(opt.max_len, dtype=np.int32)
    if len(word_freq) == 0:
        return words_np, freqs_np

    words, freqs = list(zip(*word_freq))
    words_len = min(len(words), len(words))
    for i in range(words_len):
        words_np[i] = words[i]
    for i in range(words_len):
        freqs_np[i] = freqs[i]
    return words_np, freqs_np


def most_common(data):
    return Counter(data).most_common(opt.max_len)


def baseline_label(b, m, s, d):
    return f'{b}>{m}>{s}>{d}'


def load_text(data_name='train'):
    p = Pool(opt.num_workers)
    if data_name == 'train':
        div = 'train'
        data_path_list = opt.train_data_list
    elif data_name == 'dev':
        div = 'dev'
        data_path_list = opt.dev_data_list
    elif data_name == 'test':
        div = 'test'
        data_path_list = opt.test_data_list
    else:
        assert False, '%s is not valid data name' % data_name
    product = load_data_col(data_path_list, 'product', div=div, str_decode=True)
    product = p.map(sanitize, product)
    product = p.map(hash_func_list, product)
    product = p.map(most_common, product)
    words_freqs = p.map(to_numpy, product)
    words, freqs = list(zip(*words_freqs))
    words, freqs = np.array(words), np.array(freqs)
    return words, freqs


def load_cate(data_name='train', base_label=True):
    p = Pool(opt.num_workers)
    if data_name == 'train':
        div = 'train'
        data_path_list = opt.train_data_list
    elif data_name == 'dev':
        div = 'dev'
        data_path_list = opt.dev_data_list
    elif data_name == 'test':
        div = 'test'
        data_path_list = opt.test_data_list
    else:
        assert False, '%s is not valid data name' % data_name
    b = load_data_col(data_path_list, 'bcateid', div=div)
    m = load_data_col(data_path_list, 'mcateid', div=div)
    s = load_data_col(data_path_list, 'scateid', div=div)
    d = load_data_col(data_path_list, 'dcateid', div=div)
    if not base_label:
        return b, m, s, d
    cate_label = p.starmap(baseline_label, list(zip(b, m, s, d)))
    return cate_label


if __name__ == '__main__':
    train_img_feats = load_data_col(opt.train_data_list, 'img_feat')
