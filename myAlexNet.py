from torch import nn
from torchvision import models


class MyAlexNet(nn.Module):
    """
    将AlexNet的第一个全连接层的输出作为MyAlexNet最终输出
    """
    def __init__(self):
        super(MyAlexNet, self).__init__()
        model = models.alexnet(pretrained=True)

        # Dropout and Linear
        model.classifier = model.classifier[:2]
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x
