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
Y = torch.tensor([[2.0, 1, 4, 3],
                [1, 2, 3, 4],
                [3, 8, 5, 4]])
print("X = ",X,'\n',"Y = ", Y)
#分别按照行和列尽心叠放，形成一个更大的张量
print(torch.cat((X, Y), dim=0), '\n',torch.cat((X, Y), dim=1))
print(X == Y)   #按每个元素判断是否为真

#广播机制 broadcasting mechanism
'''
由于a和b分别是和矩阵，如果让它们相加，它们的形状不匹配。 
我们将两个矩阵广播为一个更大的矩阵，如下所示：矩阵a将复制列， 
矩阵b将复制行，然后再按元素相加
'''
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)

#节省内存的操作： 在运算过程中尽量使用同一块存储区域进行数据存储
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))