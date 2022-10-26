#! /usr/bin/env python3.8
# encoding:utf-8

import base64
f=open('block_pic.png','rb') #二进制方式打开图文件
ls_f=base64.b64encode(f.read()) #读取文件内容，转换为base64编码
f.close()
print(ls_f)