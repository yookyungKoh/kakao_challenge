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

from mlp_model import MLP, HMCN, make_layer
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

train_dataset = KakaoDataset(train_path, 'train', chunk_size=40000)
valid_dataset = KakaoDataset(train_path, 'dev', chunk_size=40000)
dev_dataset = KakaoDataset(dev_path, 'dev', chunk_size=40000)

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size)
valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size)
dev_loader = DataLoader(dev_dataset, batch_size=opt.batch_size)

print('# training samples: {:,}'.format(len(train_dataset)))
print('# validation samples: {:,}'.format(len(valid_dataset)))

save_model_path = './model/checkpoints/mlp_2'
best_model_path = './model/best/mlp_2_best.pth'
result_path = './results/mlp_2_result.tsv'
continue_train = False

def get_acc(x, y):
    pred = torch.max(x,1)[1]
    y = torch.max(y,1)[1]
    correct = (pred == y).float()
    acc = torch.mean(correct)
    return acc

def get_score(b_acc, m_acc, s_acc, d_acc, i):
    bw, mw, sw, dw = 1., 1.2, 1.3, 1.4
    score = (b_acc*bw + m_acc*mw + s_acc*sw + d_acc*dw)/4
    return score/(i+1)


def main():
    opt = Option('./config.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model 
    model = HMCN(opt).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    num_params = sum([p.numel() for p in model.parameters()])
    print('Total # of params: {:,}'.format(num_params))

    if continue_train == True:
        model.load_state_dict(torch.load(best_model_path))
  
    best_loss = 100000.
    for epoch in range(opt.num_epochs):
        train(opt, train_loader, model, criterion, optimizer, epoch) 
        val_loss = evaluate(opt, valid_loader, model, criterion, make_file=False)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print('model saved at loss: %.4f'%(best_loss))
       
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), save_model_path + '_E%d.pth'%(epoch+1))
    
    model.load_state_dict(torch.load(best_model_path))
    dev_loss = evaluate(opt, dev_loader, model, criterion, make_file=True)

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

def train(opt, dataloader, model, criterion, optimizer, epoch, beta=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    model.train()
    train_loss = 0.
    acc1, acc2, acc3, acc4 = 0., 0., 0., 0.
    train_score = 0.

    for i, (inputs, targets) in enumerate(dataloader):
        targets = [t.type(torch.FloatTensor).to(device) for t in targets]
        g_targets = torch.cat((targets[0], targets[1], targets[2], targets[3]), dim=1)

        out = model(inputs)
        g_out = out[4]
        l_out = torch.cat((out[0], out[1], out[2], out[3]), dim=1)
        f_out = beta * g_out + (1-beta) * l_out

        bloss = criterion(out[0], targets[0])
        mloss = criterion(out[1], targets[1])
        sloss = criterion(out[2], targets[2])
        dloss = criterion(out[3], targets[3])    
        l_loss = bloss + mloss + sloss + dloss
        g_loss = criterion(g_out, g_targets)

        total_loss = l_loss + g_loss
        train_loss += total_loss.item()
        
        acc1 += get_acc(f_out[:,0:57], targets[0]).item()
        acc2 += get_acc(f_out[:,57:57+552], targets[1]).item()
        acc3 += get_acc(f_out[:,57+552:57+552+3190], targets[2]).item()
        acc4 += get_acc(f_out[:,57+552+3190:], targets[3]).item()
        train_score += get_score(acc1, acc2, acc3, acc4, i)

        model.zero_grad()
        total_loss.backward()
        optimizer.step()

        out_str.write('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f e-4 | Score: %.4f | b: %.4f | m: %.4f | s: %.4f | d: %.4f \r' 
                    %(epoch+1, opt.num_epochs, i+1, len(dataloader), train_loss*1e4/(i+1), train_score/(i+1), acc1/(i+1), acc2/(i+1), acc3/(i+1), acc4/(i+1)))
        out_str.flush()

    print('Training time: %.2f'%((time.time()-start_time)/ 60))

def evaluate(opt, dataloader, model, criterion, make_file=False, beta=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valid_loss = 0.
    acc1, acc2, acc3, acc4 = 0., 0., 0., 0.
    valid_score = 0.
    with torch.no_grad():
        idx = 0
        for i, (inputs, targets) in enumerate(dataloader):
            start_idx = idx
            end_idx = start_idx + len(inputs[0])
            targets = [t.type(torch.FloatTensor).to(device) for t in targets]
            g_targets = torch.cat((targets[0], targets[1], targets[2], targets[3]), dim=1)

            out = model(inputs)
            g_out = out[4]
            l_out = torch.cat((out[0], out[1], out[2], out[3]), dim=1)
            f_out = beta * g_out + (1-beta) * l_out

            bloss = criterion(out[0], targets[0])
            mloss = criterion(out[1], targets[1])
            sloss = criterion(out[2], targets[2])
            dloss = criterion(out[3], targets[3])    
            l_loss = bloss + mloss + sloss + dloss
            g_loss = criterion(g_out, g_targets)

            total_loss = l_loss + g_loss
            valid_loss += total_loss.item()
        
            acc1 += get_acc(f_out[:,0:57], targets[0]).item()
            acc2 += get_acc(f_out[:,57:57+552], targets[1]).item()
            acc3 += get_acc(f_out[:,57+552:57+552+3190], targets[2]).item()
            acc4 += get_acc(f_out[:,57+552+3190:], targets[3]).item()
            valid_score += get_score(acc1, acc2, acc3, acc4, i)

            out_str.write('Step [%d/%d] | Loss: %.4f e-4 | Score: %.4f | b: %.4f | m: %.4f | s: %.4f | d: %.4f \r' 
                        %(i+1, len(dataloader), valid_loss*1e4/(i+1), valid_score/(i+1), acc1/(i+1), acc2/(i+1), acc3/(i+1), acc4/(i+1)))
            out_str.flush()

            if make_file:
                idx = end_idx
                out_str.write('start: %d | end: %d \r' %(start_idx, end_idx))
                out_str.flush()
                create_pred_file(dev_path, start_idx, end_idx, f_out, result_path, inv_cate1)

    valid_loss /= (i+1)
    return valid_loss

def create_pred_file(data_path, start_idx, end_idx, f_out, out_path, inv_cate1):
    """
    idx : to crop pid
    out : model output (tuple)
        - out[0]: category 1 prediction (N, 57)
        - out[1]: category 2 prediction (N, 552)
        - out[2]: category 3 prediction (N, 3190)
        - out[3]: category 4 prediction (N, 404)
    """
    pid_order = []
    h = h5py.File(data_path+'data.h5py', 'r')['dev']
    pid_order.extend(h['pid'][::])
    pid_batch = pid_order[start_idx : end_idx] #(N)

    b = torch.max(f_out[:,0:57],1)[1] #(N)
    m = torch.max(f_out[:,57:57+552],1)[1]
    s = torch.max(f_out[:,57+552:57+552+3190],1)[1]
    d = torch.max(f_out[:,57+552+3190:],1)[1]
    b, m, s, d = list(map(lambda x: x+1, (b,m,s,d)))
    s[s==1] = -1
    d[d==1] = -1
    
    line = '{pid}\t{b}\t{m}\t{s}\t{d}'
    for i, pid in enumerate(pid_batch):
        assert b[i].item() in inv_cate1['b']
        assert m[i].item() in inv_cate1['m']
        assert s[i].item() in inv_cate1['s']
        assert d[i].item() in inv_cate1['d']
        
        with open(result_path, 'a') as f:
            f.write(line.format(pid=pid, b=b[i].item(), m=m[i].item(), s=s[i].item(), d=d[i].item()))
            f.write('\n')

if __name__ == '__main__':
    main()
