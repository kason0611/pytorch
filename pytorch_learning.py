# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:26:52 2021

@author: mikexia
"""

import torchvision

#准备数据集

train_data = torchvision.datasets.CIFAR10(root = '../Pytorch 啃啃啃', train = True, 
                                          transform = torchvision.transforms.ToTensor(), download = True)

test_data = torchvision.datasets.CIFAR10(root = '../Pytorch 啃啃啃', train = False, 
                                         transform = torchvision.transforms.ToTensor(), download = True)


# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集长度:{}".format(train_data_size))
print("测试集长度:{}".format(test_data_size))

# 利用DataLoader 加载数据集