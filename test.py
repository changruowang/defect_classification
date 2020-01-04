#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch
from multiprocessing import freeze_support
import cv2 as cv
from visual import imshow
from utils import *
from model_config import get_classfy_model
from glob import glob
from PIL import Image
import time

test_kind = 'gauze'
model_name = 'resnet50'
# ### densenet169   resnet18  shuffleNetV2_1x  resnet50
classes = ['defect', 'normal']
ImageBaseDir = 'H:/DateSet/{:}_defect_data/'.format(test_kind)  

# model.eval()
model = None
@torch.no_grad()
def test_on_list():
    gpu_device = torch.device('cuda:0')
    model = get_classfy_model(name=model_name, test=True, model_fold=test_kind).to(gpu_device)
    print('      name     score      kind')
    print_str = '{name:>10}{score:10.3f}{kind:>10}'
    img_path = os.path.join(ImageBaseDir, 'test', 'normal', '*')
    img_path_list = glob(img_path)
    img_names, _ = get_filenames(img_path_list)
    trans = get_transform(False)
    time_ms = 0
    for img_path, name in zip(img_path_list, img_names):
        input = trans(Image.open(img_path)).unsqueeze(0).to(gpu_device)
        starttime = time.time()
        score = model(input)
        endtime = time.time()
        time_ms += endtime - starttime
        score = torch.exp(score)/torch.sum(torch.exp(score))
        score, predictd = torch.max(score, 1)
        print(print_str.format(name=name, score=score.item(), kind=classes[predictd.item()]))
    
    print('{:}一帧图像运行时间:{:.3f}ms'.format(model_name, time_ms*1000/len(img_names)))

@torch.no_grad()
def test_on_fold():
    valid_data = torchvision.datasets.ImageFolder(os.path.join(ImageBaseDir, 'test'),
                                                    transform=get_transform(False))
    print(valid_data.class_to_idx)
    valid_dateloader = torch.utils.data.DataLoader(valid_data, batch_size=4, shuffle=False, num_workers=0)
    classify_evaluate(model, valid_dateloader, 30, gpu_device, classes=(0,1))



class classify_obj(object):
    def __init__(self):
        self.gpu_device = torch.device('cuda:0')
        self.classes = ['defect', 'normal']
        self.test_kind = 'gauze'
        #self.model = get_classfy_model(name='resnet50', test=True, model_fold= self.test_kind).to(self.gpu_device)
        self.trans = get_transform(False)
        self.model_ready = False

    def load_weight(self, name):
        self.model = get_classfy_model(name='resnet50', test=True, model_fold=name).to(self.gpu_device)
        self.model_ready = True
        self.test_kind = name
    def test_one_img(self, img_path):
        input_img = self.trans(Image.open(img_path)).unsqueeze(0).to(self.gpu_device)
        starttime = time.time()
        score = self.model(input_img)
        endtime = time.time()
        time_ms = (endtime - starttime)*1000
        score = torch.exp(score)/torch.sum(torch.exp(score))
        score, predictd = torch.max(score, 1)
        # print_str = 'score:{:.3f} {:}'.format(score, self.classes[predictd.item()])
        
        return score.item(), self.classes[predictd.item()]
    


#test_on_list()
#test_on_fold()