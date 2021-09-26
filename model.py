# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 17:00:05 2021

@author: mikexia
"""
import torch
from torch import nn


# 搭建神经网络
class model_mike(nn.Module):
    def __init__(self):
        super(model_mike, self).__init__()
        self.model = nn.Sequential(
                  nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride = 1, padding = 2),
                  nn.MaxPool2d(kernel_size = 2),
                  nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1, padding = 2),
                  nn.MaxPool2d(kernel_size = 2),
                  nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 2),
                  nn.MaxPool2d(kernel_size = 2),
                  nn.Flatten(),
                  nn.Linear(64*4*4, 64),
                  nn.Linear(64,10)            
                  )
    
    def forward(self, x):
        x = self.model(x)
        return x 


if __name__ == '__main__':
    model_pytoch = model_mike()
    input = torch.ones((64, 3, 32,32))
    output = model_pytoch(input)
    print(output.shape)