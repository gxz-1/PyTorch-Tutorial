# Finetune常用的两种方法
通常，会将模型划分为两个部分

feature extractor: 将fc层之前的部分认为是一个feature extractor  
classifier: fc层认为是classifier  

基于此，finetune大体有两种方法：

将 feature extractor部分的参数固定，冻结，不进行训练，仅训练classifier  
见代码[text](02_fintune-freeze.py)  
将 feature extractor设置较小的学习率，classifier设置较大的学习率  
见代码[text](02_fintune-multi-lr.py) 
