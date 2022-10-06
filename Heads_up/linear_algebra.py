#! /usr/bin/env python3.8
# encoding:utf-8

import torch

A = torch.arange(20).reshape(5, 4)
#计算转置
# print(A.T)
#页、行、列
X = torch.arange(24).reshape(2, 3, 4)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()   #通过分配新内存，将A的一个副本分配给B
# 矩阵按元素的乘法 哈达玛积
print(A*B)
#矩阵按照指定的axis降维
'''
默认情况下，调用求和函数会沿所有的轴降低张量的维度，
使它变为一个标量。 我们还可以指定张量沿哪一个轴来通过求和降低维度。 
以矩阵为例,为了通过求和所有行的元素来降维(轴0),
我们可以在调用函数时指定axis=0。 由于输入矩阵沿0轴降维以生成输出向量,
因此输入轴0的维数在输出形状中消失。
'''
print("Before de-dimention A:\n",A)
A_sum_axis0 = A.sum(axis=0)
print("降维0轴\n",A_sum_axis0, A_sum_axis0.shape)

#dot 点积：相同位置的按元素乘积的和
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(torch.dot(x,y))

#矩阵-向量积
print(torch.mv(A, x))
# 矩阵乘法
B = torch.ones(4,3)
print(torch.mm(A, B))

# 范数，就是表征向量的长度
#L2范数即是向量元素的和的平方根
u  = torch.tensor([3.0, -4.0])
print(torch.norm(u))
# 矩阵的F范数：矩阵元素的平方和的平方根
torch.norm(torch.ones((4, 9)))

