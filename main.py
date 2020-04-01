from sklearn.cluster import KMeans
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from visdom import Visdom

from cub2011 import Cub2011
from myAlexNet import MyAlexNet
from utils import *

batch_size = 256
n_clusters = 6

setup_seed()

train_data = Cub2011(root="dataset",
                     train=True,
                     transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
print(train_data.__len__())
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=False)

# test_data = Cub2011(root="dataset",
#                     train=False,
#                     transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
# test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

net = MyAlexNet()
# vis = Visdom()

for x, y in train_iter:
    # 将每个图像对应的feature存入csv文件
    # (batch,4096)
    features = net.forward(x)
    np.savetxt("generated_data/feature.csv", features.detach().numpy())

    # 将CNN的输出利用K-means进行聚类
    # result = KMeans(n_clusters=n_clusters, random_state=0).fit(result.detach().numpy())
    # print(result.labels_)
    # for i in range(n_clusters):
    #     indexes = np.where(result.labels_ == i)[0]
    #     images = x.index_select(0, torch.tensor(indexes[:5]))
    #     vis.images(images)
    break
