# -*- coding:utf-8 -*-
"""
@file name  : 02_containers.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-30
@brief      : 熟悉常用容器：sequential, modulelist
"""
import torch
import torch.nn as nn
from torchvision.models import alexnet


if __name__ == "__main__":
    # ==========================1. Sequential ==========================
    model = alexnet(pretrained=False)
    # alexnet中用Sequential定义了多个模块的组合
    #     self.classifier = nn.Sequential(
    #     nn.Dropout(),
    #     nn.Linear(256 * 6 * 6, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, num_classes),
    # )
    fake_input = torch.randn((1, 3, 224, 224))
    output = model(fake_input)

    # ==========================2. ModuleList ==========================

    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            #使用ModuleList生成10个线性层,不能直接用list
            self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
            # self.linears = [nn.Linear(10, 10) for i in range(10)]    # 观察model._modules，将会是空的

        def forward(self, x):
            for sub_layer in self.linears:
                x = sub_layer(x)
            return x

    model = MyModule()
    fake_input = torch.randn((32, 10))
    output = model(fake_input)
    print(output.shape)

    # ==========================3. ModuleDict ==========================
    class MyModule2(nn.Module):
        def __init__(self):
            super(MyModule2, self).__init__()
            # 使用ModuleDict定义字典类型的模型集合，按需使用
            self.choices = nn.ModuleDict({
                    'conv': nn.Conv2d(3, 16, 5),
                    'pool': nn.MaxPool2d(3)
            })
            self.activations = nn.ModuleDict({
                    'lrelu': nn.LeakyReLU(),
                    'prelu': nn.PReLU()
            })

        def forward(self, x, choice, act):
            x = self.choices[choice](x)
            x = self.activations[act](x)
            return x

    model2 = MyModule2()
    fake_input = torch.randn((1, 3, 7, 7))
    convout = model2(fake_input, "conv", "lrelu")
    poolout = model2(fake_input, "pool", "prelu")
    print(convout.shape, poolout.shape)