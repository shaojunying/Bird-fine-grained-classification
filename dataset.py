import csv
import os
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, DataLoader
import numpy as np

from net import ClusterAlexNet


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

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

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
        self.batch_size = 256
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

            net = ClusterAlexNet()
            if torch.cuda.is_available():
                net = net.cuda()
            done_nums = 0
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
                lda.fit(features, train_labels)
                features = lda.transform(features)

            result = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
            # 将结果存入csv
            # result.labels_ => the index of cluster
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

        if self.transform is not None:
            img = self.transform(img)

        return img, target


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor()
    ])
    dataset = Cub2011Cluster("dataset", cluster_id=1, transform=transform, download=True)
    print(1)

