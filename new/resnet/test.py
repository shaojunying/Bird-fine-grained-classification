import os

import torch
from flask import Flask, request
from torch import nn
from torchvision import transforms
from torchvision.datasets.folder import accimage_loader, pil_loader
from torchvision.models import resnet50, alexnet, googlenet
import pandas as pd
import torch.nn.functional as F
from werkzeug.utils import secure_filename
from flask_cors import *

app = Flask(__name__)
CORS(app, supports_credentials=True)

path = 'input.jpg'
images_classes = pd.read_csv('classes.txt', sep=' ',
                             names=['id', 'name'])


@app.route('/')
def hello_world():
    return "Hello World!"


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class MyNet(nn.Module):
    def __init__(self, n=None):
        super(MyNet, self).__init__()
        model = resnet50(pretrained=True)
        self.model1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            Flatten(),
        )
        if n is not None:
            self.model2 = nn.Linear(2048, n, bias=True)
        else:
            self.model2 = model.fc

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(path)
    if not os.path.exists(path):
        return None
    category_id, probability = get_category()
    category_name = images_classes.iloc[category_id]['name']
    # return category_name + str(probability)
    return category_name
    # return "Hahaha"


def get_category():
    """
    获取图片对应的类别
    :return:
    """
    net_path = 'model_best_76.6.pth'
    size = 224
    n_clusters = 6
    n_labels = 200
    nets = [MyNet(n_labels) for i in range(n_clusters)]
    nets = [net.eval() for net in nets]

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = default_loader(path)
    # 9
    img = transform(img)
    img = img.unsqueeze(0)

    state = torch.load(net_path)
    for net, state_dict in zip(nets, state['state_dict']):
        net.load_state_dict(state_dict)

    with torch.no_grad():
        c_s = []
        z_s = []
        for net in nets:
            z = net(img)
            z_s.append(z)
            c_s.append(z.max(dim=1).values)
        # (n,6)
        c = torch.stack(c_s, dim=1)
        # (n,6,1)
        a = c.unsqueeze(-1)
        z1 = torch.stack(z_s, dim=1)
        z = (a * z1).sum(1)
        print(z)
        z = F.softmax(z, dim=1)
        print(z)
        y_ = z.argmax(1)[0].item()
    return y_, z[0, y_].item()


if __name__ == '__main__':
    app.run()
