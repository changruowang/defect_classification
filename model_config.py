#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torchvision
import torch.nn as nn
import os
import torch
from thop import profile

CLASS_NUMS = 2

def get_classfy_model(name='resnet18', test=False, model_fold='gauze'):
    if name == 'resnet50':
        model, flag = torchvision.models.resnet50(pretrained=False), 1
    elif name == 'shuffleNetV2_1x':
        model, flag = torchvision.models.shufflenet_v2_x1_0(pretrained=False), 1   #1X
    elif name == 'densenet169':
        model, flag = torchvision.models.densenet169(pretrained=False), 3
    elif name == 'resnet18':
        model, flag = torchvision.models.resnet18(pretrained=False), 1

    if flag == 1:
        fc_fts = model.fc.in_features
        model.fc = nn.Linear(fc_fts, CLASS_NUMS)
    elif flag == 3:
        fc_fts = model.classifier.in_features
        model.classifier = nn.Linear(fc_fts, CLASS_NUMS)
    if test:
        model_param_path = os.path.join('./models/{fold:}_result/{name:}_best_model.pth'.format(name=name, fold=model_fold))
        resum_dict = torch.load(model_param_path, map_location='cpu')
        model.load_state_dict(resum_dict['model'])
        print('Success load {name:} test model!'.format(name=name))
        return  model
    print('Success load {name:} train model'.format(name=name))
   
    return model


if __name__ == "__main__":
    model_name = 'shuffleNetV2_1x'
    gpu_device = torch.device('cuda:0')
    model = get_classfy_model(name=model_name).to(gpu_device)
    model.eval()

    input_ = torch.randn(1, 3, 224, 224).to(gpu_device)
    flop, para = profile(model, inputs=(input_, )) 
    ### densenet169   resnet18  shuffleNetV2_1x  resnet50
    #print(model)
    print(model_name + ":  flop:%.2fM" % (flop/1e6), "para:%.2fM" % (para/1e6))
   