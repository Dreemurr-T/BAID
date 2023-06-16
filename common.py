'''
Common utils for training
'''

import os
import torch


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    if epoch < 40:
        lr = args.lr * (0.1 ** (epoch // 10))
    else:
        lr = args.lr * (0.1 ** 4)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(args, model, epoch):
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)