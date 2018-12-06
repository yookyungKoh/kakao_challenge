import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
import sys

from mlp_model import MLP
from utils import KakaoDataset
from misc import Option, get_logger

opt = Option('./config.json')
out_str = sys.stdout

# Load data

load_time = time.time()
data_root = './data/'
train_path = os.path.join(data_root, 'train/')
dev_path = os.path.join(data_root, 'dev/')

train_dataset = KakaoDataset(train_path, chunk_size=40000)
dev_dataset = KakaoDataset(dev_path, chunk_size=40000)

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size)
valid_loader = DataLoader(dev_dataset, batch_size=opt.batch_size)

save_model_path = './model/checkpoints/'
best_model_path = './model/best/'

def get_acc(x, y):
    pred = torch.max(x,1)[1]
    y = torch.max(y,1)[1]
    correct = (pred == y).float()
    acc = torch.mean(correct)
    return acc

def main():
    opt = Option('./config.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model 
    model = MLP(opt).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    
    best_loss = 100000.
    for epoch in range(opt.num_epochs):
        print('Training..........')
        train(opt, train_loader, model, criterion, optimizer, epoch)
        print('Validating..........')
        val_loss = evaluate(opt, valid_loader, model, criterion, epoch)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path + '_E%d.h5'%(epoch+1))
            print('model saved at loss: %.4f'%(best_loss))

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), save_model_path + '_E%d.h5'%(epoch+1))

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

def evaluate(opt, dataloader, model, criterion, epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    valid_loss = 0.
    acc1, acc2, acc3, acc4 = 0., 0., 0., 0.
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
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

            out_str.write('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Acc: %.4f | cate1: %.4f | cate2: %.4f | cate3: %.4f | cate4: %.4f \r' 
                        %(epoch+1, opt.num_epochs, i+1, len(dataloader), valid_loss/(i+1), valid_acc/(i+1), acc1/(i+1), acc2/(i+1), acc3/(i+1), acc4/(i+1)))
            out_str.flush()
            
    return valid_loss

if __name__ == '__main__':
    main()
