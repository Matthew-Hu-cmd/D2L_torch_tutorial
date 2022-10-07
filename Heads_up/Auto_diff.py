#! /usr/bin/env python3.8
# encoding:utf-8

import torch

x = torch.arange(4.0)
x.requires_grad(True)   #即是x=torch.arange(4.0,requires_grad=True)
# 表示即是后续要储存住x的梯度
# 构造 y = XT ·X
y  = 2 * torch.dot(x, x)
y.backward()    #调用反向传播函数
