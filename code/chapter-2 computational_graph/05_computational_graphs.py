# -*- coding:utf-8 -*-
"""
@file name  : 05_computational_graphs.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-06
@brief      : 计算图中的叶子结点观察
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd.function import Function

if __name__ == "__main__":
    import torch

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)  # retain_grad()
    y = torch.mul(a, b)

    # 在计算过程中动态搭建计算图，同时针对每个tensor存储计算梯度必备的grad_fn
    #  调用backward时自动反向传播计算计算图中所有梯度（自动求导机制）
    y.backward() 
    print(w.grad)

    # 查看叶子结点 
    # w,x作为运算的开始是叶子节点，梯度得到保留
    # 其他变量xsby作为中间变量不是叶子节点，梯度在方向传播后不保留
    print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
    # 查看梯度
    print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)
    # 查看 grad_fn
    print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)
