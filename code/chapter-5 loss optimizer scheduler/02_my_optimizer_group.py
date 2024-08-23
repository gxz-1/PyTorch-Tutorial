import torch
import torch.nn as nn
import torch.optim as optim

#------------------参数组使用的实例代码-------------------------

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化网络
model = SimpleNet()

# 定义优化器，使用两个参数组
optimizer = optim.SGD([
    {'params': model.fc1.parameters(), 'lr': 0.01},
    {'params': model.fc2.parameters(), 'lr': 0.001}
])

# 一个简单的训练循环
for epoch in range(100):
    # 输入随机数据
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)

    # 前向传播
    outputs = model(inputs)
    loss = torch.mean((outputs - targets) ** 2)

    # 反向传播和优化
    optimizer.zero_grad() #梯度清零
    loss.backward() #计算梯度
    optimizer.step() #参数优化

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
