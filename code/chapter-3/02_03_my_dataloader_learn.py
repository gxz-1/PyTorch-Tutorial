import os
import torch
from torch.utils.data import Dataset,DataLoader,ConcatDataset,Subset
from PIL import Image
from torchvision.transforms import transforms

class ImageNet(Dataset):
    label_map={"ants":0,"bees":1}
    my_transform=None

    def __init__(self,dataset_dir,type) -> None:
        self.dataset_list=[]
        self.dataset_dir = dataset_dir
        self.type = type # train/val
        # trainsforms对数据集进行预处理
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.my_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
        ])
        #读取数据
        self.readDataset()

    def readDataset(self):
        # dir：当前目录 subdirs：dir的子目录 files：dir的子文件 
        root_dir = os.path.join(self.dataset_dir,self.type)
        for dir,subdirs,files in os.walk(root_dir):
            if dir != root_dir: #遍历子目录由subdirs进行，dir不再进行
                break
            for subdir in subdirs:
                label=self.label_map[subdir]
                for _,_,imgs in os.walk(os.path.join(dir,subdir)):
                    for img in imgs:
                        img_dir=os.path.join(dir,subdir,img)
                        self.dataset_list.append((img_dir,label))

    #获取数据集项目总数，提供给DataLoader
    def __len__(self):
        return len(self.dataset_list)
    
    #映射式（还有种迭代式）根据index获取数据集单个项目，提供给DataLoader
    def __getitem__(self, index):
        img_dir,label=self.dataset_list[index]
        img = Image.open(img_dir)
        if self.my_transform is not None:
            img = self.my_transform(img)
        return img,label

if __name__=="__main__":
    # 从磁盘加载数据集Dataset
    train_dataset=ImageNet("dataset/mini-hymenoptera_data","train")
    val_dataset=ImageNet("dataset/mini-hymenoptera_data","val")
    # Dataset类实现__getitem__和__len__方法提供给Dataloader加载数据
    # num_workers：cpu加载数据的核心数
    # 通常加载使用cpu，在训练循环中，可以在获取批量数据后使用to(device移动到 GPU 上
    train_loader=DataLoader(train_dataset,batch_size=2,shuffle=True,num_workers=6)
    val_loader=DataLoader(val_dataset,batch_size=2,shuffle=True,drop_last=True,num_workers=6)
    #查看第i个batch的数据size
    for i,(img,label) in enumerate(train_loader):
        print("train",i,img.size(),label.size())
    for i,(img,label) in enumerate(val_loader):
        print("val",i,img.size(),label.size())


    #2.对于不同的数据集，可以通过ConcatDataset类合并
    #val_dataset可以是其他数据集，只要__getitem__返回值一致
    all_dataset = ConcatDataset([train_dataset, val_dataset]) 
    #3.Subset获取子数据集
    all_sub_set = Subset(all_dataset, [0, 1, 2, 5])  # 将这4个样本抽出来构成子数据集