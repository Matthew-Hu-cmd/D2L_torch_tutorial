#! /usr/bin/env python3.8
# encoding:utf-8

import torch

x = torch.arange(12)    #创建一个一维数组0-11
# print(x.shape) 

X = x.reshape(3, 4)     #不改变元素的值的情况，改变张量形状
# print(X)
torch.zeros((3, 3, 4))  #创建全0与全1
torch.ones((3, 3, 4))   #三面，三行四列

#列表赋值
X1 = torch.tensor(
    [
        [2, 1, 4, 3],
        [1, 1, 1, 1],
        [2, 7, 2, 0]
    ]
)
# print(X1)

###常见标准运算（按元素进行运算）**是按元素求y次幂
x = torch.tensor([1.0, 3.4, 2, 1])
y = torch.tensor([1, 2, 2, 1])
# print(x+y,'\n', x-y,'\n', x*y,'\n', x/y,'\n', x**y)

# 多个张量结合
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([],[],[])