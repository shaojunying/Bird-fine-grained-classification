from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from visdom import Visdom

from dataset import Cub2011
from net import ClusterAlexNet
from utils import *

batch_size = 256
n_clusters = 6
n_components = 128


def save_feature():
    """
    使用预训练的CNN网络得到每个图像的feature,并存入文件
    :return:
    """
    train_data = Cub2011(root="dataset",
                         train=True,
                         transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    net = ClusterAlexNet()
    if torch.cuda.is_available():
        net = net.cuda()
    done_nums = 0
    with open("generated_data/feature.csv", "w") as f:
        for x, y in train_iter:
            # 将每个图像对应的feature存入csv文件
            # (batch,4096)
            if torch.cuda.is_available():
                x = x.cuda()
            features = net.forward(x)
            np.savetxt(f, features.cpu().detach().numpy())
            done_nums += x.shape[0]
            print(str(done_nums) + "/" + str(len(train_data)))


def cluster(use_lda=False):
    """
    按照提取出的特征将样本进行K-means聚类
    :return:
    """
    with open('generated_data/feature.csv', 'r') as f:
        data = np.loadtxt(f)
    print("Loaded data successfully!!")

    if use_lda:
        # 这里对所有feature使用LDA进行降维
        train_data = Cub2011(root="dataset",
                             train=True,
                             transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
        train_labels = np.array([label for _, label in train_data])
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(data, train_labels)
        data = lda.transform(data)

    result = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    for i in range(n_clusters):
        indexes = np.where(result.labels_ == i)[0]
        if use_lda:
            path = "generated_data/cluster_with_LDA/" + str(i) + ".csv"
        else:
            path = "generated_data/cluster/" + str(i) + ".csv"
        with open(path, "w") as f:
            np.savetxt(f, indexes)
        print("Saved the {} cluster".format(i))


def show_result(use_lda=False):
    train_data = Cub2011(root="dataset",
                         train=True,
                         transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    vis = Visdom()
    for i in range(n_clusters):
        if use_lda:
            path = "generated_data/cluster_with_LDA/" + str(i) + ".csv"
        else:
            path = "generated_data/cluster/" + str(i) + ".csv"
        with open(path, "r") as f:
            data = np.loadtxt(f)
        n_samples = 5
        samples = None
        # indices = random.sample(data.tolist(), n_samples)
        indices = data.tolist()[-6:-1]
        for index in indices:
            data = train_data[int(index)][0].expand([1, 3, 256, 256])
            if samples is None:
                samples = data
            else:
                samples = torch.cat((samples, data), dim=0)
        vis.images(samples)


def main():
    setup_seed(2)
    # save_feature()
    # cluster(use_lda=True)
    show_result(use_lda=False)


if __name__ == '__main__':
    main()
