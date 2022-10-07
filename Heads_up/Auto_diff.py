#! /usr/bin/env python3.8
# encoding:utf-8

import torch

x = torch.arange(4.0)
x.requires_grad(True)   #即是x=torch.arange(4.0,requires_grad=True)
# 表示即是后续要储存住x的梯度
# 构造 y = XT ·X    说白了对于一个列向量x而言这个就是他的平方
y  = 2 * torch.dot(x, x)
y.backward()    #调用反向传播函数,求导
print(x.grad)   #可以用grad来访问梯度
# pytorch会累积梯度，所以每次重新计算要清零
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
# 张量的求导计算一般来说都是标量，不会涉及到向量对向量求导
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u #相当于就最后结果用u表示

'''
即使构建函数的计算图需要通过Python控制流
（例如，条件、循环或任意函数调用），
我们仍然可以计算得到的变量的梯度。 在下面的代码中，
while循环的迭代次数和if语句的结果都取决于输入a的值。
'''

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)