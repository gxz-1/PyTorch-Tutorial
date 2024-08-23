# Tensorboard 基础与使用

## tensorboard 安装

```bash
conda install tensorboard
pip install setuptools==58.0.0  
```

## tensorboard 初体验

1.使用tensoboard的SummarWriter生成evevts文件，用于web端可视化  
2.指定events文件所在路径，启动tensorboard  
```bash
tensorboard --logdir log/dir --port 6007 
```

## torchinfo 

torchinfo，可用一键实现模型参数量计算、各层特征图形状计算和计算量计算等功能  
torchinfo 主要提供了一个函数summary打印模型参数数量等信息  
具体见05_model_print.py  
```bash
pip install torchinfo
```