import os

import torch
from flask import Flask, request
from torch import nn
from torchvision import transforms
from torchvision.datasets.folder import accimage_loader, pil_loader
from torchvision.models import resnet50, alexnet, googlenet
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def hello_world():
    return "Hello World!"

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class MyNet(nn.Module):
    def __init__(self, n = None):
        super(MyNet, self).__init__()
        model = alexnet(pretrained=True)
        self.model1 = nn.Sequential(
            model.features,
            model.avgpool,
            Flatten(),
            nn.Sequential(
                *(model.classifier[:-1])
            )
        )
        if n is None:
            self.model2 = model.classifier[-1]
        else:
            self.model2 = nn.Linear(4096, n, bias=True)

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x



path = 'input.jpg'
net_path = 'new/alexnet/checkpoint.pth'
size = 224
n_clusters = 6
n_labels = 200
nets = [MyNet(n_labels) for i in range(n_clusters)]


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
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = default_loader(path)
    img = transform(img)
    img = img.unsqueeze(0)
    state = torch.load(net_path, map_location='cpu')
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
        y_ = z.argmax(1)
    print(z.sort(1))
    print(y_)
    return "Hahaha"


if __name__ == '__main__':
    # app.run()
    pass

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

state = torch.load(net_path, map_location=lambda storage, loc: storage)
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
    y_ = z.argmax(1)
print(z.sort(1))
print(y_)
