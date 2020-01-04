#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from PIL import Image, ImageDraw
import matplotlib.pylab as plt
from torchvision.transforms.transforms import ToPILImage
import numpy as np
import torchvision
from utils import *
import os

label_map = {0: 'defect', 1: 'normal'}

def imshow(inp, labels=None):
    title = []
    if labels is not None:
        for i in range(labels.shape[0]):
            print(i)
            title.append(label_map[labels[i].item()])

    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=20)
    plt.axis('off')  #去掉坐标轴
    plt.show()  # pause a bit so that plots are updated
    
if __name__ == "__main__":
    ImageBaseDir = 'H:/DateSet/screw_defect_data/'
    valid_data = torchvision.datasets.ImageFolder(os.path.join(ImageBaseDir, 'eval'),
                                                  transform=get_transform(False))
    valid_dateloader = torch.utils.data.DataLoader(valid_data, batch_size=4, shuffle=True, num_workers=0)
    images, labels = iter(valid_dateloader).next()
    imshow(torchvision.utils.make_grid(images), labels)
