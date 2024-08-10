import os
import torch
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# 案例1：将图片大小resize
my_transforms=transforms.Resize((500,500))
for root,_,imgs_dir in os.walk("dataset/air"):
    if root != "dataset/air": #不遍历子目录
        break
    for img_dir in imgs_dir:
        img=Image.open(os.path.join(root,img_dir))
        resize_img=my_transforms(img)
        resize_img.save("dataset/air/resize/"+img_dir)

# 案例2
transforms_func = transforms.Compose([
    transforms.Resize((500,500)),
    # 中心裁剪出224x224的区域
    transforms.CenterCrop(224),
    # ToTensor操作：1.通道右移由(H*W*C)变为(C*H*W) 2.数值除以255缩放到0到1之间 3.转换为tensor
    transforms.ToTensor(),
    # 对于RGB三个通道分别归一化，如R通道以0.485均值0.229方差标准化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
for root,_,imgs_dir in os.walk("dataset/air"):
    if root != "dataset/air": #不遍历子目录
        break
    for img_dir in imgs_dir:
        img=Image.open(os.path.join(root,img_dir))
        trans_img=transforms_func(img)
        print("before:{} {}  after:{}".format(img.size,img.mode,trans_img.size()))

# 案例3：fiveCorp 从每个角和中心裁剪出五个224x224的图像
five_crop = transforms.Compose([
    transforms.FiveCrop(size=(224, 224)),  # 定义裁剪大小
    # 转换每个裁剪后的图像为张量
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])) 
])
# 应用 FiveCrop
for root,_,imgs_dir in os.walk("dataset/air"):
    if root != "dataset/air": #不遍历子目录
        break
    for img_dir in imgs_dir:
        img=Image.open(os.path.join(root,img_dir))
        crops=five_crop(img)
        print(crops.size()) #[5, 3, 224, 224]
