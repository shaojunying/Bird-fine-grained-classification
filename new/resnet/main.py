import csv
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.models import resnet50

size = 224
lr_decay = 40

batch_size = 4
n_clusters = 6
n_components = 128
n_labels = 200
n_epochs = 1000
lr = 1e-2
train_dependently_directory = F"checkpoint/independently"
train_jointly_directory = F"checkpoint/jointly"
dataset_directory = F"../../dataset"


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


def exist_checkpoint(directory, filename='checkpoint.pth'):
    return os.path.exists(os.path.join(directory, filename))


def save_checkpoint(state, directory, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    # directory = "runs/%s/%s/%s/"%(config.dataset, config.model, config.checkname)

    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth'))


def load_checkpoint_1(directory, filename='checkpoint.pth'):
    filename = os.path.join(directory, filename)
    state = None
    if os.path.exists(filename):
        state = torch.load(filename)
    return state


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


class Cub2011Cluster(Dataset):

    def __init__(self, root, cluster_id=None, train=True, transform=None, loader=default_loader, download=True):
        super(Cub2011Cluster, self).__init__()
        self.root = root
        self.cluster_id = cluster_id
        self.train = train
        self.transform = transform
        self.loader = loader
        self.download = download
        self.batch_size = 32
        self.dataset = Cub2011(root, train, transform, loader, download)
        self.data_iter = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        # print(self.load_feature("features.csv"))
        if self.train:
            self.data = self.load_cluster_info(6)

            if cluster_id is not None:
                self.data = self.data[self.data.cluster_id == cluster_id]
        else:
            self.data = self.dataset.data[self.dataset.data.is_training_img == 0]

    def load_feature(self, filename):
        """
        Load the feature of images
        :return:
        """
        if not os.path.exists(os.path.join(self.root, filename)):

            net = MyNet()
            if torch.cuda.is_available():
                net = net.cuda()
            done_nums = 0
            self.dataset = Cub2011(self.root, self.train, transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]), self.loader, self.download)
            self.data_iter = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
            with open(os.path.join(self.root, filename), "w") as f:
                print("Start to compute the features")
                writer = csv.writer(f)
                for x, y in self.data_iter:
                    if torch.cuda.is_available():
                        x = x.cuda()
                    # (batch,4096)
                    features = net.forward(x)
                    writer.writerows(features.cpu().detach().numpy())
                    done_nums += x.shape[0]
                    print("Finished proportion:", str(done_nums) + "/" + str(len(self.dataset)))
                print("Finished the compute of features")

        # read features from csv file
        features = []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                features += [[float(i) for i in row]]
        return np.array(features)

    def load_cluster_info(self, n_clusters, use_lda=True):
        """
        Load the information of cluster
        :return:
        """
        path = os.path.join(self.root, "cluster.csv")
        if not os.path.exists(path):
            n_components = 128
            features = self.load_feature("features.csv")

            if use_lda:
                # 这里对所有feature使用LDA进行降维
                dataset = self.dataset
                train_labels = dataset.data[dataset.data.is_training_img == 1].target.values
                train_labels = train_labels.reshape(-1, 1)
                lda = LinearDiscriminantAnalysis(n_components=n_components)
                print(features.shape, train_labels.shape)
                lda.fit(features, train_labels)
                features = lda.transform(features)

            result = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
            # 将结果存入csv
            # result.labels_ => the index of cluster
            # with open('kmean.pickle', 'wb') as f:
            #     pickle.dump(result, f)
            # with open('lda.pickle', 'wb') as f:
            #     pickle.dump(lda, f)
            print(result)
            cluster_ids = result.labels_.reshape(-1, 1).tolist()
            id = [i for i in range(1, len(cluster_ids) + 1)]
            cluster_ids = np.insert(cluster_ids, 0, id, axis=1)
            with open(path, mode='w') as f:
                writer = csv.writer(f)
                writer.writerows(cluster_ids)
        # Load cluster from csv
        data_cluster = pd.read_csv(os.path.join(self.root, 'cluster.csv'),
                                   names=['img_id', 'cluster_id'])
        data = self.dataset.data.merge(data_cluster, on='img_id')
        return data

    def __len__(self):
        return self.data.values.shape[0]

    def __getitem__(self, idx):

        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.dataset.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        img = img.crop((sample.x, sample.y, sample.x + sample.width, sample.y + sample.height))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

best_acc = 0
if __name__ == '__main__':
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

    datasets = [Cub2011Cluster(root=dataset_directory, cluster_id=i, train=True, transform=train_transform)
                for i in range(n_clusters)]
    data_loaders = [DataLoader(dataset, shuffle=True, batch_size=batch_size) for dataset in datasets]

    train_set = Cub2011(root=dataset_directory, train=True, transform=train_transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    test_set = Cub2011Cluster(root=dataset_directory, train=False, transform=test_transform)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    val_set = Cub2011(root=dataset_directory, train=True, transform=test_transform)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size)

    nets = [MyNet(n_labels) for i in range(n_clusters)]
    nets = [net.cuda() for net in nets]
    criterion = nn.CrossEntropyLoss()


    def train_with_clustering(epoch):
        for data_loader, net in zip(data_loaders, nets):

            optimizer = torch.optim.SGD([{'params': net.model1.parameters(), 'lr': 0},
                                         {'params': net.model2.parameters(), 'lr': lr}],
                                        lr=lr, momentum=0.9, weight_decay=1e-4)

            train_loss, total, correct = 0, 0, 0
            for batch_id, (x, y) in enumerate(data_loader):
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                y_ = net(x)
                loss = criterion(y_, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += y.shape[0]
                correct += (y_.argmax(dim=1) == y).sum().cpu().item()
                train_loss += loss.item()
                if batch_id % 8 == 0:
                    pass
                print("Epoch:{}, {}/{},train loss:{},train acc:{}"
                      .format(epoch, batch_id, len(data_loader),
                              train_loss / (batch_id + 1), correct / total))


    def train_jointly(epoch):
        temp = []
        if epoch < 11:
            for net in nets:
                temp.append({'params': net.model1.parameters(), 'lr': 0})
                temp.append({'params': net.model2.parameters(), 'lr': lr})
        elif epoch < 26:
            for net in nets:
                temp.append({'params': net.model1.parameters(), 'lr': lr * 0.1})
                temp.append({'params': net.model2.parameters(), 'lr': lr})
        else:
            for net in nets:
                temp.append({'params': net.model1.parameters(), 'lr': lr * 0.1})
                temp.append({'params': net.model2.parameters(), 'lr': lr * 0.1})

        optimizer = torch.optim.SGD(temp, lr=lr, momentum=0.9, weight_decay=1e-4)
#         optimizer = torch.optim.Adam(temp, lr=lr)
        correct, total, train_loss = 0, 0, 0
        for batch_id, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            c_s = []
            z_s = []
            for net in nets:
                z = net(x)
                z_s.append(z)
                c_s.append(z.max(dim=1).values)
            # (n,6)
            c = torch.stack(c_s, dim=1)
            # (n,6,1)
            a = F.softmax(c, dim=1).unsqueeze(-1)

            z1 = torch.stack(z_s, dim=1)
            # (n, 200)
            z = (a * z1).sum(1)
            loss = criterion(z, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # (n)
            y_ = z.argmax(1)
            correct += (y_ == y).sum().cpu().item()
            total += y.shape[0]
            if batch_id % 8 == 0:
                pass
            print("Epoch:{},train batch:{}/{}, loss:{} ,acc:{}".format(epoch, batch_id, len(train_loader),
                                                                       train_loss / (batch_id + 1), correct / total))


    def test_with_clustering(epoch, directory):
        global best_acc
        global nets
        correct, total = 0, 0
        with torch.no_grad():
            nets = [net.eval() for net in nets]
            for batch_id, (x, y) in enumerate(test_loader):
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                c_s = []
                z_s = []
                for net in nets:
                    z = net(x)
                    z_s.append(z)
                    c_s.append(z.max(dim=1).values)
                # (n,6)
                c = torch.stack(c_s, dim=1)
                # (n,6,1)
                # a = F.softmax(c, dim=1).unsqueeze(-1)
                a = c.unsqueeze(-1)
                z1 = torch.stack(z_s, dim=1)
                z = (a * z1).sum(1)
                y_ = z.argmax(1)
                correct += (y_ == y).sum().cpu().item()
                total += y.shape[0]
                if batch_id % 8 == 0:
                    pass
                print("Epoch:{},test batch:{}/{},acc:{}".format(epoch, batch_id, len(test_loader), correct / total))
        acc = correct / total
        is_best = False
        if acc > best_acc:
            is_best = True
            best_acc = acc

        # 保存模型
        # save_checkpoint({
        #     'epoch': epoch,
        #     'state_dict': [net.state_dict() for net in nets]
        # }, directory, is_best)

        print("Epoch:{},test acc:{}(best:{})".format(epoch, acc, best_acc))


    start = 0
    has_joint_checkpoint = exist_checkpoint(train_jointly_directory)
    # if not has_joint_checkpoint:
    #     print("Start train dependently")
    #     # 这里需要专家网络首先单独训练30轮
    #     # 读取最新
    #     state = load_checkpoint_1(train_dependently_directory)
    #     if state is not None:
    #         start = state['epoch'] + 1
    #         for net, state_dict in zip(nets, state['state_dict']):
    #             net.load_state_dict(state_dict)
    #     for epoch in range(start, 46):
    #         train_with_clustering(epoch)
    #         if epoch % 2 == 0:
    #             test_with_clustering(epoch, train_dependently_directory)
    # # 需要专家网络放在一起进行训练
    # print("Start train jointly")
    # start = 0
    # # 读取最优
    # best_filename = os.path.join(train_jointly_directory, 'model_best.pth')
    # if os.path.exists(best_filename):
    #     state = torch.load(best_filename)
    #     for net, state_dict in zip(nets, state['state_dict']):
    #         net.load_state_dict(state_dict)
    #     test_with_clustering(state['epoch'], train_jointly_directory)
    #
    # 读取最新
    if True:
        # state = load_checkpoint_1(train_jointly_directory)
        state = torch.load("model_best_76.6.pth")
        start = state['epoch'] + 1
        for net, state_dict in zip(nets, state['state_dict']):
            net.load_state_dict(state_dict)

    # state = torch.load('ori.pth')
    # for net in nets:
    #     net.load_state_dict(state)

    for epoch in range(start, 100):
        # train_jointly(epoch)
        test_with_clustering(epoch, train_jointly_directory)
