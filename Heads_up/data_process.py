#! /usr/bin/env python3.8
# encoding:utf-8
import os
import pandas as pd
import torch
#使用pandas处理原始数据，并转换为张量格式
#创建逗号分隔值文件csv作为人工数据集
os.makedirs(os.path.join('.', 'data'), exist_ok=True)  #将会在运行目录下生成文件夹
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)
#可以注意到有不少地方的数据缺失：处理方法：插值、删除
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
#分为输入、输出两类
inputs = inputs.fillna(inputs.mean())   #数值阈用平均值填充
print(inputs)
'''
由于无法正确的用平均值表示类型
使用0 1值来实现增维判断
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
'''

inputs = pd.get_dummies(inputs, dummy_na=True) 
print(inputs)
#最终转换成张量
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y