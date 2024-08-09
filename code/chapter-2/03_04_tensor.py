import numpy as np
import torch

# torch.tensor()可以从numpy中创建
data1=np.array([[1,2,3],[4,5,6]])
# 参数列表中*之后的参数仅限关键字指定，如dtype、device
t=torch.tensor(data1,dtype=torch.float64,device="cuda:1") 
print(t)

# from_numpy的方式共享一块内存
t=torch.from_numpy(data1)
data1[0][0]=100
print(t)

#按照数值创建
torch.zeros((5,33),device="cuda:0")
torch.ones((5,20),dtype=torch.int32)
torch.full((5,10),3.1415926)
torch.arange(1,20,2) #1到20步长为2生成
torch.linspace(0,1,20) #0到1均匀生成20个数
print(torch.eye(3,4))  #单位对角矩阵

#按照概率分布创建
mean=2 #均值
std=1 #方差
torch.normal(mean,std,(10,1)) #在mean和std的高斯分布上采样10个数
torch.rand((3,4)) #0到1均匀分布
torch.randint(1,10,(3,4)) #1到10整数均匀分布

#scater_方法
src=torch.arange(1,11,dtype=torch.float32).reshape((2,5))
print(src)
dst=torch.zeros((5,5))
print(dst.scatter_(dim=0,index=torch.tensor([[0,1,3,2,2]]),src=src))
# 1.将src中的元素替换dst中的元素
# 2.如何确定依次替换哪个元素？
# dim和index结合表示：替换第[0,1,3,2,2]行的元素
# 3.行确定了如何确定列？
# 除dim外的其他维度按照indx对应维度的索引确定
# 4.src中的元素如何取？
# 按照index的索引取
# 综上，dst中第[0,1,3,2,2]行第[0,1,2,3,4]列的元素被src的[0,0][0,1][0,2][0,3][0,4]替换

index=torch.tensor([[0,2,4],[1,2,3]])
dst=torch.zeros((5,5))
print(dst.scatter_(dim=1,index=index,src=src))
#dst中第[0,2,4]列第0行 第[1,2，3]列第1行的元素
#依次被src中[0,0][0,1][0,2] [1,0][1,1][1,2]的元素替换