#! /usr/bin/env python3.8
# encoding:utf-8

import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

#生成数据集：初始化一个具有基本线性特征的数据集
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w))) #随机数抽取，输出(num_examples, len(w))的张量
#这里的X是吧所有的x[x1, x2, x3,...]存成了按照行来堆叠的形式，自然w有多少，x就有多少个对应的角标
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) #加入均值为0，标准差为0.001的噪声
    return X, y.reshape((-1, 1))
    #reshape中-1表示自动计算，即根据其他维度信息计算出-1处应该是多少->此处即是把y变成一个列向量

true_w = torch.tensor([2, -3.4])
#这就是在指定要生成一个[2, -3.4]这个张量
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

#查看生成的数据  
# plt.plot(features[:,1], labels, 'bo', label = 'Labeled data') #绘制第二个特征与label相关关系图
# plt.title('Scatter dot graph')
# plt.legend() # 添加图例
# plt.savefig("/root/D2L_torch_tutorial/Linear_neural_network/Linear_regression/scatter_dot_graph.png")

'''
读取小批量的数据
in: 批大小、特征、标签
out: 被打乱后提取出的单个样本
'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))#创建对应的下标
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):    #每次跳跃batch_size的大小
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])  #有可能会跳出最大值
        yield features[batch_indices], labels[batch_indices]    #对应的随机数下标的特征和标签
        # yield就是return一个数值，并且记住这个返回的位置，下次继续从这个位置返回

'''
小批量训练
初始化模型参数
'''

batch_size = 10

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
#requires_grad  反向传播时，有这个属性的tensor会自动求导

# 定义模型
def linreg(X, w, b):    #@save
    # 线性回归模型
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """
    小批量随机梯度下降
    params是一个含有w、b的list
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size   #自动求导时，梯度会存在.grad里面
            param.grad.zero_()

#with 的用法：适合资源需要合理访问的场合，确保不管使用过程中是否异常都会有清理操作释放资源

'''
训练函数
超参数：
步长
'''
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
