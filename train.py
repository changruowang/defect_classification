#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch
from multiprocessing import freeze_support
from visual import imshow
from utils import *
from tensorboardX import SummaryWriter
import argparse
from model_config import get_classfy_model

# def get_args():
#     parser = argparse.ArgumentParser(description='My detection code training based on pytorch!')

#     parser.add_argument('--data_path', default='/mnt2/changruowang/data/PennFudanPed', help='dataset path')
#     # parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
#     # parser.add_argument('--dataset', default='coco', help='dataset')
#     parser.add_argument('--device', default='cuda:0', help='device')
#     parser.add_argument('--b', '--batch_size', default=2, type=int)
#     parser.add_argument('--epochs', default=100, type=int, metavar='N',
#                         help='number of total epochs to run')
#     parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
#                         help='number of data loading workers (default: 8)')
#     parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
#     parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                         help='momentum')
#     parser.add_argument('--wd', default=0.0005, type=float,dest='wd',
#                          help='weight decay (default: 0.0005)')
#     parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
#     # parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
#     # parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
#     # parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
#     parser.add_argument('--resume', default=None, help='resume from checkpoint')
#     parser.add_argument('--test_only', action='store_true', dest='test_only', help='test only', default=False)
#     parser.add_argument('--log_dir', default='./log_dir', help='path where to save')
#     args = parser.parse_args()
#     return args


start_lr = 0.001
milestones = [20, 80, 200, 700]
model_name = 'shuffleNetV2_1x'   
# ImageBaseDir = "/mnt/wufangbu_data/"
#ImageBaseDir = "/mnt/changruowang/data/gauze_defect_data/"  #纱布
ImageBaseDir = "/mnt/changruowang/data/screw_defect_data/"  #螺丝
resum_model_name ='epoch_now_model.pth'
EN_RESUM = False
NUM_EPOCHES = 1500
EVAL_EVERY_EPOCHS = 20
BATCH_SZIE = 32

def classification_train():
    # args = get_args()
    cuda_device = torch.device("cuda:2")
    log_dir = './log_dir'
    if log_dir:
        mkdir(log_dir)
    tf_logger = SummaryWriter(log_dir)
 
    ####模型准备
    model = get_classfy_model(name=model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    ####数据集准备

    train_data = torchvision.datasets.ImageFolder(os.path.join(ImageBaseDir, 'train'),
                                                  transform=get_transform(True))
    valid_data = torchvision.datasets.ImageFolder(os.path.join(ImageBaseDir, 'eval'),
                                                  transform=get_transform(False))
    train_dateloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SZIE, shuffle=True, 
                                                    num_workers=4,  drop_last=True)
    valid_dateloader = torch.utils.data.DataLoader(valid_data, batch_size=4, shuffle=False, num_workers=4)

    ####Resum
    best_score = 0
    start_epoch = 0 
    model.to(cuda_device)
    resum_model_path = os.path.join('./models/', resum_model_name)
    if os.path.exists(resum_model_path) and EN_RESUM:
        print('training start from args.resume...')
        resum_dict = torch.load(resum_model_path)
        model.load_state_dict(resum_dict['model'])
        optimizer.load_state_dict(resum_dict['optimizer'])
        lr_scheduler.load_state_dict(resum_dict['lr_scheduler'])
        start_epoch = resum_dict['epoch']
        best_score = resum_dict['best_pr_score']
    ####
    print('Start classify training!')
    metric_logger = MetricLogger(delimiter='  ', logger=tf_logger)
    metric_logger.add_meter('lr', window_size=1, fmt='{value:.6f}')

    for epoch in range(start_epoch, NUM_EPOCHES+1):
        model.train()
        for images, labels in metric_logger.log_every(train_dateloader, 2, epoch):
            images, labels = images.to(cuda_device), labels.to(cuda_device)

            out = model(images)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        lr_scheduler.step()
       
        if (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
            classify_result = classify_evaluate(model, valid_dateloader, print_feq=2,
                                            device=cuda_device, classes=(0, 1))
            now_score = classify_result['accuracy']       ###用准确率评估                           
            if now_score >= best_score:
                best_score = now_score
                save_checkpoint({'best_pr_score': now_score,
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'lr_scheduler': lr_scheduler.state_dict(),
                                    'epoch': epoch}, './models/epoch_best_model.pth')
            else:
                save_checkpoint({'best_pr_score': now_score,
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'lr_scheduler': lr_scheduler.state_dict(),
                                    'epoch': epoch}, './models/epoch_now_model.pth')

    print('End classify training!')



if __name__ == '__main__':
    classification_train()
    # ImageBaseDir = "/mnt/wufangbu_data/"
    # train_data = torchvision.datasets.ImageFolder(os.path.join(ImageBaseDir, 'train'),
    #                                               transform=get_transform(True))
    # valid_data = torchvision.datasets.ImageFolder(os.path.join(ImageBaseDir, 'eval'),
    #                                               transform=get_transform(False))
    # train_dateloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=8)
    # valid_dateloader = torch.utils.data.DataLoader(valid_data, batch_size=4, shuffle=False, num_workers=8)
    #
    # images, labels = iter(train_dateloader).next()
    #
    #
    # print(images.shape, labels)
    #
    # imshow(torchvision.utils.make_grid(images), labels)



 
