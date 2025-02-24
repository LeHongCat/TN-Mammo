#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:48:45 2022

@author: raju
"""
import argparse
import os
import time
import torch
import torchvision.transforms as transforms
import yaml
from munch import Munch
from torch.utils.tensorboard import SummaryWriter
from models import MV_DEFEAT_ddsm
from dataset import DDSM_dataset, ThongNhat_dataset, VinDr_dataset
from utils import save_checkpoint, AverageMeter, accuracy
from criterions import EDL_CE_Loss
from tqdm import tqdm

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("On which device we are on:{}".format(device))


# accuracy = MulticlassAccuracy(num_classes=3)
def adjust_learning_rate(optimizer, epoch):
    lr = cfg.optimizer.lr
    for e in cfg.optimizer.lr_decay_epochs:
        if epoch >= e:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    model.train()
    end = time.time()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f'Epoch {epoch} [TRAIN]')
    for i, (input1, input2, input3, input4, target) in pbar:
        data_time.update(time.time() - end)
        input1 = input1.to(device)
        input2 = input2.to(device)
        input3 = input3.to(device)
        input4 = input4.to(device)
        target = target.to(device)
        output = model(input1, input2, input3, input4)
        loss = criterion(output, target)

        acc = accuracy(output, target)
        losses.update(loss.item(), input1.size(0))
        accuracies.update(acc, input1.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix({
            'Time': f'{batch_time.val:.3f} ({batch_time.avg:.3f})',
            'Loss': f'{losses.val:.4f} ({losses.avg:.4f})',
            'Acc': f'{accuracies.val:.4f} ({accuracies.avg:.4f})'
        })

    return batch_time.avg, data_time.avg, losses.avg, accuracies.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.eval()
    end = time.time()
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                desc='[VALID]')
    with torch.no_grad():
        for i, (input1, input2, input3, input4, target) in pbar:
            input1 = input1.to(device)
            input2 = input2.to(device)
            input3 = input3.to(device)
            input4 = input4.to(device)
            target = target.to(device)
            output = model(input1, input2, input3, input4)
            loss = criterion(output, target)

            acc = accuracy(output, target)
            losses.update(loss.item(), input1.size(0))
            accuracies.update(acc, input1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({
                'Time': f'{batch_time.val:.3f} ({batch_time.avg:.3f})',
                'Loss': f'{losses.val:.4f} ({losses.avg:.4f})',
                'Acc': f'{accuracies.val:.4f} ({accuracies.avg:.4f})'
            })

    return batch_time.avg, losses.avg, accuracies.avg


def main(cfg):
    if cfg.training.resume is not None:
        log_dir = cfg.training.log_dir
        checkpoint_dir = os.path.dirname(cfg.training.resume)
    else:
        log_dir = os.path.join(cfg.training.logs_dir, 
                              '{}_{}'.format(cfg.data.dataset_name, cfg.data.task),
                              cfg.training.fusion_type,
                              cfg.data.analysis,
                              f'fold_{cfg.fold}')
        checkpoint_dir = os.path.join(cfg.training.checkpoints_dir,
                                    '{}_{}'.format(cfg.data.dataset_name, cfg.data.task),
                                    cfg.training.fusion_type,
                                    cfg.data.analysis,
                                    f'fold_{cfg.fold}')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    print('log_dir: {}'.format(log_dir))
    print('checkpoint_dir: {}'.format(checkpoint_dir))

    model = MV_DEFEAT_ddsm(cfg).to(device)

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = EDL_CE_Loss(cfg)
    # criterion = CE_loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    start_epoch = 0
    if cfg.training.resume is not None:
        if os.path.isfile(cfg.training.resume):
            print("=> loading checkpoint '{}'".format(cfg.training.resume))
            checkpoint = torch.load(cfg.training.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.training.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.training.resume))
            print('')
            raise Exception

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train__valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ])

    if cfg.data.dataset_name == 'DDSM':
        train_dataset = DDSM_dataset(cfg.data.root, view_laterality=cfg.data.laterality, 
                                   split=f'fold{cfg.fold}_train', transform=train__valid_transform)
        val_dataset = DDSM_dataset(cfg.data.root, view_laterality=cfg.data.laterality, 
                                 split=f'fold{cfg.fold}_val', transform=train__valid_transform)
    elif cfg.data.dataset_name == 'ThongNhat':
        train_dataset = ThongNhat_dataset(cfg.data.root, view_laterality=cfg.data.laterality, 
                                        split=f'fold{cfg.fold}_train', transform=train__valid_transform)
        val_dataset = ThongNhat_dataset(cfg.data.root, view_laterality=cfg.data.laterality, 
                                      split=f'fold{cfg.fold}_val', transform=train__valid_transform)
    elif cfg.data.dataset_name == 'VinDr':
        train_dataset = VinDr_dataset(cfg.data.root, view_laterality=cfg.data.laterality, 
                                    split=f'fold{cfg.fold}_train', transform=train__valid_transform)
        val_dataset = VinDr_dataset(cfg.data.root, view_laterality=cfg.data.laterality, 
                                  split=f'fold{cfg.fold}_val', transform=train__valid_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
                                               num_workers=cfg.data.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
                                             num_workers=cfg.data.workers, pin_memory=True)

    train_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    best_accuracy = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        lr = adjust_learning_rate(optimizer, epoch)
        train_summary_writer.add_scalar('learning_rate', lr, epoch + 1)

        train_batch_time, train_data_time, train_loss, train_accuracy = train(train_loader, model, criterion, optimizer,
                                                                              epoch)
        train_summary_writer.add_scalar('batch_time', train_batch_time, epoch + 1)
        train_summary_writer.add_scalar('loss', train_loss, epoch + 1)
        train_summary_writer.add_scalar('accuracy', train_accuracy, epoch + 1)

        val_batch_time, val_loss, val_accuracy = validate(val_loader, model, criterion)
        val_summary_writer.add_scalar('batch_time', val_batch_time, epoch + 1)
        val_summary_writer.add_scalar('loss', val_loss, epoch + 1)
        val_summary_writer.add_scalar('accuracy', val_accuracy, epoch + 1)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_path = save_checkpoint(checkpoint_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
            }, filename='model_best.pth')
        
        checkpoint_path = save_checkpoint(checkpoint_dir, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
        }, filename='model_latest.pth')
        
        cfg.training.log_dir = log_dir
        cfg.training.resume = checkpoint_path
        with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
            f.write(cfg.toYAML())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='ThongNhat_config.yml')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (0-4) for cross validation')
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        cfg = Munch.fromYAML(f)
    cfg.fold = args.fold  # Add fold number to config
    main(cfg)
