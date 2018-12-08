import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
import sys
import h5py
import json
import six

from mlp_model import MLP
from utils import KakaoDataset
from misc import Option, get_logger

opt = Option('./config.json')
out_str = sys.stdout

def get_inverted_cate1(cate1):
    inv_cate1 = {}
    for d in ['b', 'm', 's', 'd']:
        inv_cate1[d] = {v: k for k, v in six.iteritems(cate1[d])}
    return inv_cate1

# Load data
cate1 = json.loads(open('../cate1.json').read())
inv_cate1 = get_inverted_cate1(cate1)

load_time = time.time()
data_root = './data/'
train_path = os.path.join(data_root, 'train/')
dev_path = os.path.join(data_root, 'dev/')

train_dataset = KakaoDataset(train_path, chunk_size=40000)
dev_dataset = KakaoDataset(dev_path, chunk_size=40000)

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size)
valid_loader = DataLoader(dev_dataset, batch_size=opt.batch_size)

save_model_path = './model/checkpoints/bi_mlp'
best_model_path = './model/best/bi_mlp'
result_path = './results/bi_mlp_result.tsv'

def get_acc(x, y):
    pred = torch.max(x,1)[1]
    y = torch.max(y,1)[1]
    correct = (pred == y).float()
    acc = torch.mean(correct)
    return acc

def main():
    opt = Option('./config.json')
    opt.num_epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model 
    model = MLP(opt).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    best_loss = 100000.
#    for epoch in range(opt.num_epochs):
#        train(opt, train_loader, model, criterion, optimizer, epoch)
#        
#        if val_loss < best_loss:
#            best_loss = val_loss
#            torch.save(model.state_dict(), best_model_path + '_E%d.pth'%(epoch+1))
#            print('model saved at loss: %.4f'%(best_loss))
#        
#        if (epoch+1) % 1 == 0:
#            torch.save(model.state_dict(), save_model_path + '_E%d.pth'%(epoch+1))
    
    model.load_state_dict(torch.load(save_model_path+'_E1.pth'))
    evaluate(opt, valid_loader, model, criterion)

    pid_order = []
    h = h5py.File('./data/dev/data.h5py', 'r')['dev']
    pid_order.extend(h['pid'][::])
    no_ans = '{pid}\t-1\t-1\t-1\t-1'
    with open(result_path, 'r') as f:
        file_len = len(f.readlines())
        print('total prediction length:', file_len)
    with open(result_path, 'a') as f:
        pid_none = pid_order[file_len:]
        for pid in pid_none:
            f.write(no_ans.format(pid=pid))
            f.write('\n')
    print('created file at %s'%(result_path))

def train(opt, dataloader, model, criterion, optimizer, epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    model.train()
    train_loss = 0.
    acc1, acc2, acc3, acc4 = 0., 0., 0., 0.
    for i, (inputs, targets) in enumerate(dataloader):
        targets = [t.type(torch.FloatTensor).to(device) for t in targets]
        out = model(inputs)

        bloss = criterion(out[0], targets[0])
        mloss = criterion(out[1], targets[1])
        sloss = criterion(out[2], targets[2])
        dloss = criterion(out[3], targets[3])
        total_loss = bloss + mloss + sloss + dloss
        train_loss += total_loss.item()
        
        acc1 += get_acc(out[0], targets[0]).item()
        acc2 += get_acc(out[1], targets[1]).item()
        acc3 += get_acc(out[2], targets[2]).item()
        acc4 += get_acc(out[3], targets[3]).item()
        train_acc = (acc1+acc2+acc3+acc4)/4
        
        model.zero_grad()
        total_loss.backward()
        optimizer.step()

        out_str.write('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Acc: %.4f | cate1: %.4f | cate2: %.4f | cate3: %.4f | cate4: %.4f \r' 
                    %(epoch+1, opt.num_epochs, i+1, len(dataloader), train_loss/(i+1), train_acc/(i+1), acc1/(i+1), acc2/(i+1), acc3/(i+1), acc4/(i+1)))
        out_str.flush()

    print('Training time: %.2f'%((time.time()-start_time)/ 60))

def evaluate(opt, dataloader, model, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valid_loss = 0.
    acc1, acc2, acc3, acc4 = 0., 0., 0., 0.
    
    with torch.no_grad():
        idx = 0
        for i, (inputs, targets) in enumerate(dataloader):
            start_idx = idx
            end_idx = start_idx + len(inputs[0])
            targets = [t.type(torch.FloatTensor).to(device) for t in targets]
            out = model(inputs)
            
            bloss = criterion(out[0], targets[0])
            mloss = criterion(out[1], targets[1])
            sloss = criterion(out[2], targets[2])
            dloss = criterion(out[3], targets[3])
            total_loss = bloss + mloss + sloss + dloss
            valid_loss += total_loss.item()

            acc1 += get_acc(out[0], targets[0]).item()
            acc2 += get_acc(out[1], targets[1]).item()
            acc3 += get_acc(out[2], targets[2]).item()
            acc4 += get_acc(out[3], targets[3]).item()
            valid_acc = (acc1+acc2+acc3+acc4)/4

#            out_str.write('Step [%d/%d] | Loss: %.4f | Acc: %.4f | cate1: %.4f | cate2: %.4f | cate3: %.4f | cate4: %.4f \r' 
#                        %(i+1, len(dataloader), valid_loss/(i+1), valid_acc/(i+1), acc1/(i+1), acc2/(i+1), acc3/(i+1), acc4/(i+1)))
#            out_str.flush()
            
            idx = end_idx
            out_str.write('start: %d | end: %d \r' %(start_idx, end_idx))
            out_str.flush()

            create_pred_file(start_idx, end_idx, out, result_path, inv_cate1)

def create_pred_file(start_idx, end_idx, out, out_path, inv_cate1):
    """
    idx : to crop pid
    out : model output (tuple)
        - out[0]: category 1 prediction (N, 57)
        - out[1]: category 2 prediction (N, 552)
        - out[2]: category 3 prediction (N, 3190)
        - out[3]: category 4 prediction (N, 404)
    """
    pid_order = []
    h = h5py.File('./data/dev/data.h5py', 'r')['dev']
    pid_order.extend(h['pid'][::])
    pid_batch = pid_order[start_idx : end_idx] #(N)

    b = torch.max(out[0],1)[1] #(N)
    m = torch.max(out[1],1)[1]
    s = torch.max(out[2],1)[1]
    d = torch.max(out[3],1)[1]
#    b, m, s, d = list(map(lambda x: x+1, (b,m,s,d)))

    line = '{pid}\t{b}\t{m}\t{s}\t{d}'
    for i, pid in enumerate(pid_batch):
        with open(result_path, 'a') as f:
                        
            assert b[i] in inv_cate1['b']
            assert m[i] in inv_cate1['m']
            assert s[i] in inv_cate1['s']
            assert d[i] in inv_cate1['d']

            b = inv_cate1['b'][b[i]]
            m = inv_cate1['b'][m[i]]
            s = inv_cate1['b'][s[i]]
            d = inv_cate1['b'][d[i]]

            f.write(line.format(pid=pid, b=b, m=m, s=s, d=d))
            f.write('\n')

if __name__ == '__main__':
    main()
