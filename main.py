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
    # show_result(use_lda=False)


if __name__ == '__main__':
    main()
