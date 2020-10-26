import os

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.models import alexnet

filename = "checkpoint.pth"
best_filename = "model_best.pth"

size = 256
lr_decay = 40

batch_size = 32
n_clusters = 6
n_components = 128
n_labels = 200
n_epochs = 1000
train_dependently_directory = F"independently"
train_jointly_directory = F"jointly"
dataset_directory = F"dataset"
lr = 1e-2


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        boxes = pd.read_table(os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'), sep=' ',
                              names=['img_id', 'x', 'y', 'width', 'height'])
        self.boxes = boxes.to_numpy()

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        self.data = self.data.merge(boxes, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        img = img.crop((sample.x, sample.y, sample.x + sample.width, sample.y + sample.height))

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class MyNet(nn.Module):
    def __init__(self, n):
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


train_transform = transforms.Compose([
    transforms.Resize((int(size * 1.25), int(size * 1.25))),
    transforms.RandomRotation(15),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

net = MyNet(200)

net = net.cuda()

train_set = Cub2011(root=dataset_directory, train=True, transform=train_transform)
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

test_set = Cub2011(root=dataset_directory, train=False, transform=test_transform)
test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

# optimizer = torch.optim.SGD([{'params': net.parameters()}], lr=lr, momentum=0.9)
optimizer = torch.optim.SGD([{'params': net.model1.parameters(), 'lr': 0},
                             {'params': net.model2.parameters(), 'lr': lr}], lr=lr, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

best_acc = 0
if os.path.exists(best_filename):
    state = torch.load(best_filename)
    net.load_state_dict(state)
    with torch.no_grad():
        correct, total, train_loss = 0, 0, 0
        for batch_id, (x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()
            y_ = net(x)
            correct += (y_.argmax(1) == y).sum().cpu().item()
            total += y.shape[0]
            acc = correct / total
            best_acc = acc
            print("best_acc:{}".format(best_acc))

if os.path.exists(filename):
    state = torch.load(filename)
    net.load_state_dict(state)
for epoch in range(100):
    # train
    correct, total, train_loss = 0, 0, 0
    for batch_id, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        y_ = net(x)
        loss = criterion(y_, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (y_.argmax(1) == y).sum().cpu().item()
        total += y.shape[0]
        train_loss += loss.cpu().item()
        print("Epoch:{},train batch:{}/{}, loss:{} ,acc:{}".format(epoch, batch_id, len(train_loader),
                                                                   train_loss / (batch_id + 1), correct / total))

    with torch.no_grad():
        correct, total, train_loss = 0, 0, 0
        for batch_id, (x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()
            y_ = net(x)
            correct += (y_.argmax(1) == y).sum().cpu().item()
            total += y.shape[0]
            acc = correct / total
            print("Epoch:{},test batch:{}/{}, loss:{} ,acc:{}(best:{})".format(epoch, batch_id, len(train_loader),
                                                                               train_loss / (batch_id + 1),
                                                                               correct / total, best_acc))
    if acc > best_acc:
        torch.save(net.state_dict(), best_filename)
        best_acc = acc
    torch.save(net.state_dict(), filename)
