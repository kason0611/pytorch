# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:26:52 2021

@author: mikexia
"""



# git上传(先确定PYTHON PATH)
!git remote rm origin

#把这个文件夹变成Git可管理的仓库。
!git init

#把该目录下的所有文件添加到仓库
!git add "pytorch_learning.py"
!git add "model.py"

#把项目提交到仓库。
!git commit -m 'upload' 

#(将本地仓库与GitHub上创建好的目标远程仓库进行关联。 …后面加的是GitHub目标仓库地址)。
!git remote add origin https://github.com/kason0611/pytorch.git

#把本地库的所有内容推送到GitHub远程仓库上。
!git push -u origin master





!git pull origin master --allow-unrelated-histories
#!git config --global user.name "kason0611"
#!git config --global user.email kason0611@gmail.com






!git push -u origin master


import torch 

import torchvision

from torch.utils.data import DataLoader

from torch import nn

from model import *

from torch.utils.tensorboard import SummaryWriter


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

# 利用 DataLoader 加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64) 

# 搭建神经网络
model_pytorch = model_mike()

# 损失函数
loss_fn = nn.CrossEntropyLoss() 

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(params = model_pytorch.parameters(), lr = learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数 
total_train_step = 0 

# 记录测试的次数
total_test_step = 0 

# 训练的轮数
epoch = 2

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i+1))
    
    # 训练步骤开始
    for data in train_dataloader:
        imgs,targets = data 
        outputs = model_pytorch(imgs)
        loss = loss_fn(outputs, targets)
        
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar('train loss', loss.item(), total_train_step+1)
   
    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0 
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data  
            outputs = model_pytorch(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item() 
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = accuracy + total_accuracy
            total_test_step = total_test_step + 1 
            
    print("整体测试loss：{}".format(total_test_loss))      
    print("整体测试accuracy：{}".format(total_accuracy/test_data_size))      
  
    writer.add_scalar('test loss', total_test_loss, total_test_step+1)
    writer.add_scalar('test accuracy', total_accuracy/test_data_size, total_test_step+1)  
    
    torch.save(model_pytorch, "model_pytorch.pth".format(i))
    print("模型{}已保存".format(i))
    
writer.close()