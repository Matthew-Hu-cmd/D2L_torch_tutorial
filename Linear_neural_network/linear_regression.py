#! /usr/bin/env python3.8
# encoding:utf-8
import numpy as np
import torch
from torch.utils import data

from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)#用d2l的合成线性模型的函数来生成数据集

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """
    构造一个PyTorch数据迭代器:
    1. 将features、labels给转化成为 tensorDataset
    2. DataSet拿到数据集之后,用dataLoader这个方法每次随机的从中选取数据
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)  #最终得到数据迭代器
'''
iter()函数用于实现一个迭代器
next()函数用于返回迭代器的下一个项目
'''
next(iter(data_iter))
#神经网络模型
net = nn.Sequential(nn.Linear(2, 1))    #输入为2 输出为1
# Sequential这个容器可以理解为这是一个List of layers 直接明了的定义了整个网络的结构

#下面初始化参数，net[0]就是第0层的相关参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
# mean square error loss均方误差，十分常见

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()    #pytorch已经自动将各个维度的误差求和了
        trainer.step()  #step函数进行模型的更新
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)