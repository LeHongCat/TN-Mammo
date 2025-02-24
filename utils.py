# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:39:00 2022

@author: rajgudhe
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from itertools import cycle
from sklearn.metrics import auc, roc_curve

def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        # Đảm bảo output và target ở định dạng phù hợp
        _, pred = output.max(1)
        target = target.long()
        correct = pred.eq(target).sum().item()
        return correct * 100.0 / target.size(0)


def save_checkpoint(dir, state, epoch, is_best=False):
    filepath = os.path.join(dir, 'checkpoint_{}.pth'.format(epoch))
    torch.save(state, filepath)
    
    if is_best:
        # Lưu một bản copy là best model
        best_filepath = os.path.join(dir, 'model_best.pth')
        torch.save(state, best_filepath)
    
    return filepath

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count