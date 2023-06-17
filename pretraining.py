import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import PretrainingDataset
from model import ResNetPretrain
import argparse
import torch.optim as optim
from pretraining_utils.pretrain_losses import cal_deg_loss, cal_trp_loss
from common import *
import time
import torchvision.transforms as transforms
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=100)

    # when to activate trp loss
    parser.add_argument('--init_epoch', type=int, default=30)
    # deg loss weight
    parser.add_argument('--deg_weight', type=float, default=1.0)
    # trp loss weight
    parser.add_argument('--trp_weight', type=float, default=0.1)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoint/ResNet_Pretrain')

    parser.add_argument('--save_freq', type=int,
                        default=2)

    return parser.parse_args()


def train(args):
    path_csv = 'pretraining_utils/train.csv'
    opt_csv = 'pretraining_utils//operations.csv'

    dataset = PretrainingDataset(path_csv=path_csv, opt_csv=opt_csv)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = ResNetPretrain(num_classes=30).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)

    for epoch in range(args.epoch):
        model.train()
        for step, train_data in enumerate(train_loader):
            optimizer.zero_grad()

            deg_loss = 0.0
            trp_loss = 0.0

            original_image = train_data[0].to(device)
            opt = train_data[1]
            degraded_image = train_data[2].to(device)
            trp_1 = train_data[3].to(device)
            trp_2 = train_data[4].to(device)

            if epoch < args.init_epoch:
                _, logits = model(degraded_image)
                deg_loss += cal_deg_loss(args, opt, logits)
            else:
                original_features, _ = model(original_image)
                degraded_features, logits = model(degraded_image)
                trp1_features, _ = model(trp_1)
                trp2_features, _ = model(trp_2)

                deg_loss += cal_deg_loss(args, opt, logits)
                trp_loss += cal_trp_loss(args, original_features, degraded_features, trp1_features, trp2_features, opt)

            total_loss = args.deg_weight * deg_loss + args.trp_weight * trp_loss

            total_loss.backward()
            optimizer.step()

            print("Epoch: %3d Step: %5d / %5d deg_loss: %.8f trp_loss: %.8f total_loss: %.8f" % (epoch, step,
                      len(train_loader), deg_loss, trp_loss, total_loss))

        adjust_learning_rate(args, optimizer, epoch)
        
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(args, model, epoch)


if __name__ == '__main__':
    args = parse_args()
    train(args)
