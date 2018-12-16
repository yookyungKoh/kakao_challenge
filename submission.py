import os
import json
import threading

import fire
import h5py
import tqdm
import numpy as np
import pandas as pd
import six

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from six.moves import zip, cPickle

from misc import get_logger, Option
from network import TextOnly, top1_acc

DEV_DATA_LIST = ['../dev.chunk.01']


def write_prediction_result(self, data, pred_y, meta, out_path, readable):
    pid_order = []
    DEV_DATA_LIST = ['../dev.chunk.01']
    for data_path in DEV_DATA_LIST:
        h = h5py.File(data_path, 'r')['dev']
        pid_order.extend(h['pid'][::])
    pid_order = [x.decode('utf-8') for x in pid_order]
    mlp_pred = pd.read_csv('./mlp_1_result.tsv', sep='\t', index_col=0, header=None)
    mlp_pred_ordered = mlp_pred.loc[mlp_pred.index.intersection(pid_order)].reindex(pid_order)
    mlp_pred_ordered = mlp_pred_ordered.astype(int)
    mlp_pred_ordered.to_csv('result.tsv', sep='\t', header=None)
