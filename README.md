# Bird Fine-grained Classification Based on Subset and Deep Learning
## 基于子类和深度学习的细粒度鸟类识别

### 项目基本思路
- 使用预训练的CNN模型得到每个图像的特征，将训练结束后每个图像对应的feature进行存储
- 将所有图像的feature进行K-means聚类，这里的K值设置为6
- 创建K个CNN网络模型，将属于i类别的图像放入第i个模型进行训练

### 实验结果
AlexNet-tl: 41.6%(51.78%)
GoogLeNet-tl: (69.9344%)
ResNet-tl: 77.5975%(77.7011%)