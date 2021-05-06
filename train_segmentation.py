import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import warnings
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import argparse
import pandas as pd
from utils import *
from dataloader import PolypDataset
from seg_model import DarkraNet

# fix random seeds for reproducibility
SEED = 2021
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

criterion = nn.BCELoss()
def train(args, model, train_loader, val_loader, optimizer):

    print('\n-----> Start Training Loop...')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    t_mIoU = []
    t_loss = []
    v_mIoU = []
    v_loss = []
    max_dice = 0.
    size_rates = [0.75, 1, 1.25]    # multi-scale training
    # size_rates = [1]
    for epoch in range(args.start_epoch, args.max_epoch+1):
        model.train()
        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        print('Epoch {}/{}'.format(epoch, args.max_epoch))

        t_loss_record = 0.
        t_mIoU_record = 0.
        for idx, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                data, mask = pack
                data, mask = data.to(DEVICE), mask.to(DEVICE)

                trainsize = int(round(352*rate/32)*32)    # rescale
                if rate != 1:
                    data = F.upsample(data, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    mask = F.upsample(mask, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                lateral_map_5, lateral_map_3, lateral_map_2 = model(data)
                pred = nn.Sigmoid()(lateral_map_2)

                loss5 = structure_loss(lateral_map_5, mask)
                loss3 = structure_loss(lateral_map_3, mask)
                loss2 = structure_loss(lateral_map_2, mask)
                loss = loss2 + loss3 + loss5

                loss.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()

                if rate == 1:
                    t_loss_record += loss
                    t_mIoU_record += metrics(pred, mask)[0]
                    loss_record2.update(loss2.data, args.batch_size)
                    loss_record3.update(loss3.data, args.batch_size)
                    loss_record5.update(loss5.data, args.batch_size)

            if idx % args.log_interval == 0:
                print('Train Epoch: {}  [{}/{} ({:.0f}%)] IoU: {:.4f} Dice: {:.4f} Loss: {:.4f}'
                    '\n[lateral-2: {:.4f}, lateral-3: {:.4f}, lateral-5: {:.4f}]'
                    .format(epoch, idx*len(data), len(train_loader.dataset),
                    100.*idx/len(train_loader), metrics(pred, mask)[0], metrics(pred, mask)[1], loss.item(),
                    loss_record2.show(), loss_record3.show(), loss_record5.show()))

        t_mIoU.append(t_mIoU_record / (idx+1))
        t_loss.append(t_loss_record / (idx+1))

        if epoch % args.save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path,
                       'seg_{}.pth'.format(epoch)))

        val_mIoU, val_loss, val_dice = val(args, model, val_loader)
        v_mIoU.append(val_mIoU)
        v_loss.append(val_loss)

        if val_dice > max_dice:
            max_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.save_path,
                       'seg_best_{}_{:.4f}.pth'.format(epoch, val_dice)))

    print('-----> Done !')
    learning_curve(t_mIoU, t_loss, v_mIoU, v_loss)



def val(args, model, val_loader):

    model.eval()
    v_loss = 0.
    v_mIoU = 0.
    v_dice = 0.
    v_f2 = 0.
    v_p = 0.
    v_r = 0.
    with torch.no_grad():
        for idx, (image, mask) in enumerate(val_loader):
            data, mask = image.to(DEVICE), mask.to(DEVICE)

            lateral_map_5, lateral_map_3, lateral_map_2 = model(data)
            pred = nn.Sigmoid()(lateral_map_2)

            loss5 = structure_loss(lateral_map_5, mask)
            loss3 = structure_loss(lateral_map_3, mask)
            loss2 = structure_loss(lateral_map_2, mask)
            loss = loss2 + loss3 + loss5

            v_loss += loss
            v_mIoU += metrics(pred, mask)[0]
            v_dice += metrics(pred, mask)[1]
            v_p += metrics(pred, mask)[2]
            v_r += metrics(pred, mask)[3]
            v_f2 += metrics(pred, mask)[4]

    v_mIoU /= (idx+1)
    v_loss /= (idx+1)
    v_dice /= (idx+1)
    v_f2 /= (idx+1)
    v_p /= (idx+1)
    v_r /= (idx+1)
    print('\n***Val mIoU: {:.4f} mDice: {:.4f} F2: {:.4f} Precision: {:.4f} Recall: {:.4f}***\n'.format(
        v_mIoU, v_dice, v_f2, v_p, v_r))


    return v_mIoU, v_loss, v_dice



def parse_args():
    parser = argparse.ArgumentParser(description='Polyp segmentation')
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--save_epoch', default=5, type=int)
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--save_path', default='./saved_models', type=str, help='Model checkpoint path')
    parser.add_argument('--data_dir', default='./Polyp_dataset', type=str, help='Polyp data directory')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    trainset = PolypDataset(data_path=args.data_dir, csv='train.csv', mode='Test')
    valset = PolypDataset(data_path=args.data_dir, csv='val.csv', mode='Test')
    testset = PolypDataset(data_path=args.data_dir, csv='test.csv', mode='Test')

    print('# images in trainset:', len(trainset))   # Should print 1490
    print('# images in valset:', len(valset))       # Should print 186
    print('# images in testset:', len(testset))     # Should print 188

    # Use the torch dataloader to iterate through the dataset
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    model = DarkraNet()

    if args.start_epoch != 1:
        check = torch.load(os.path.join(args.save_path, 'seg_{}.pth'.format(args.start_epoch)))
        model.load_state_dict(check)
    # check = torch.load(os.path.join(args.save_path, 'DarkraNet_best.pth'))
    # model.load_state_dict(check)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, train_loader, val_loader, optimizer)
