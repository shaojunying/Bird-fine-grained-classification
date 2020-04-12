from torch import nn
from torchvision import models


class ClusterAlexNet(nn.Module):
    """
    用于聚类的AlexNet
    将AlexNet的第一个全连接层的输出作为 ClusterAlexNet 最终输出
    """

    def __init__(self):
        super(ClusterAlexNet, self).__init__()
        model = models.alexnet(pretrained=True)

        # Dropout and Linear
        model.classifier = model.classifier[:2]
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x


class ClassificationAlexNet(nn.Module):
    """
    用于分类的AlexNet
    将AlexNet的最后一层替换成新的线性回归星
    """

    def __init__(self, n):
        """
        :param n: 分类的目标类别数
        """
        super(ClassificationAlexNet, self).__init__()
        model = models.alexnet(pretrained=True)
        model.classifier = model.classifier[:-1]
        model.classifier.add_module(str(len(model.classifier)), nn.Linear(4096, n))
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x
