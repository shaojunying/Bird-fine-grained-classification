import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
# from visdom import Visdom
import torch.nn.functional as F

from dataset import Cub2011, Cub2011Cluster
from net import ClassificationAlexNet
# from utils import *
from utils import setup_seed, load_checkpoint_1, save_checkpoint, adjust_learning_rate, exist_checkpoint

batch_size = 64
n_clusters = 6
n_components = 128
n_labels = 200
n_epochs = 1000
lr = 0.01
train_dependently_directory = F"/content/drive/My Drive/checkpoint/independently"
train_jointly_directory = F"/content/drive/My Drive/checkpoint/jointly"

# def show_result(use_lda=False):
#     train_data = Cub2011(root="dataset",
#                          train=True,
#                          transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
#     vis = Visdom()
#     for i in range(n_clusters):
#         if use_lda:
#             path = "generated_data/cluster_with_LDA/" + str(i) + ".csv"
#         else:
#             path = "generated_data/cluster/" + str(i) + ".csv"
#         with open(path, "r") as f:
#             data = np.loadtxt(f)
#         n_samples = 5
#         samples = None
#         # indices = random.sample(data.tolist(), n_samples)
#         indices = data.tolist()[-6:-1]
#         for index in indices:
#             data = train_data[int(index)][0].expand([1, 3, 256, 256])
#             if samples is None:
#                 samples = data
#             else:
#                 samples = torch.cat((samples, data), dim=0)
#         vis.images(samples)
# net = ClassificationAlexNet(n_labels)
# if torch.cuda.is_available():
#     net = net.cuda()
# entropy = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=lr)
# transform = transforms.Compose([
#     transforms.Resize((256, 256)), transforms.ToTensor()
# ])
#
best_test_acc = 0
best_val_acc = 0


def test_using_alexnet(epoch):
    test_data = Cub2011(root="dataset", train=False, transform=transform)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    global best_test_acc
    correct, test_loss = 0, 0
    with torch.no_grad():
        for test_x, test_y in test_iter:
            test_x, test_y = test_x.cuda(), test_y.cuda()
            test_y_ = net(test_x)
            test_loss += entropy(test_y_, test_y)
            correct += (test_y_.argmax(dim=1) == test_y).sum().cpu().item()
    test_acc = correct / len(test_data)
    best_test_acc = max(test_acc, best_test_acc)

    print("Epoch:{},test loss:{}, test acc:{}(best: {})"
          .format(epoch, test_loss, test_acc, best_test_acc))


def main():
    setup_seed(2)

    # show_result(use_lda=False)
    # nets = [ClassificationAlexNet(n_labels) for _ in range(n_clusters)]

    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor()
    ])

    datasets = [Cub2011Cluster(root="dataset", cluster_id=i, train=True, transform=transform)
                for i in range(n_clusters)]
    # datasets = [Cub2011(root="dataset",  train=True, transform=transform)
    #             for i in range(n_clusters)]
    datasets_len = [len(dataset) for dataset in datasets]
    data_loaders = [DataLoader(dataset, shuffle=True, batch_size=batch_size) for dataset in datasets]

    train_set = Cub2011(root="dataset", train=True, transform=transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    test_set = Cub2011Cluster(root="dataset", train=False, transform=transform)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    val_set = Cub2011(root="dataset", train=True, transform=transform)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size)

    nets = [ClassificationAlexNet(n_labels) for i in range(n_clusters)]
    nets = [nn.DataParallel(net) for net in nets]
    nets = [net.cuda() for net in nets]
    criterion = nn.CrossEntropyLoss()

    def train_with_clustering(epoch):
        max_dataset_len = max(datasets_len)
        for dataset_len, data_loader, net in zip(datasets_len, data_loaders, nets):
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            adjust_learning_rate(optimizer, epoch)
            train_loss, total, correct = 0, 0, 0
            for _ in range(0, max_dataset_len, dataset_len):
                for batch_id, (x, y) in enumerate(data_loader):
                    x, y = x.cuda(), y.cuda()
                    y_ = net(x)
                    loss = criterion(y_, y)
                    train_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += y.shape[0]
                    correct += (y_.argmax(dim=1) == y).sum().cpu().item()

                    print("Epoch:{}, {}/{},train loss:{},train acc:{}"
                          .format(epoch, batch_size * (batch_id + 1), dataset_len,
                                  train_loss / (batch_id + 1), correct / total))

    def train_jointly(epoch):
        optimizer = torch.optim.SGD([{'params': net.parameters()} for net in nets], lr=lr)

        correct, total, train_loss = 0, 0, 0
        for batch_id, (x, y) in enumerate(train_loader):
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
            # (n)
            y_ = z.argmax(1)
            loss = criterion(z, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            correct += (y_ == y).sum().cpu().item()
            total += y.shape[0]
            print("Epoch:{}, batch:{}/{}, loss:{} ,acc:{}".format(epoch, batch_id, len(train_loader), train_loss / (batch_id + 1), correct / total))

    def val_with_clustering(epoch):
        global best_val_acc
        correct, total = 0, 0
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(val_loader):
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
                z = (a * z1).sum(1)
                y_ = z.argmax(1)
                correct += (y_ == y).sum().cpu().item()
                total += y.shape[0]
                print("batch:{}/{},acc:{}".format(batch_id, len(val_loader), correct / total))
        acc = correct / total
        best_val_acc = max(best_val_acc, acc)
        print("Epoch:{},val acc:{}(best:{})".format(epoch, acc, best_val_acc))

    def test_with_clustering(epoch, directory):
        global best_test_acc
        correct, total = 0, 0
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(test_loader):
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
                z = (a * z1).sum(1)
                y_ = z.argmax(1)
                correct += (y_ == y).sum().cpu().item()
                total += y.shape[0]
                print("Epoch:{}, batch:{}/{},acc:{}".format(epoch, batch_id, len(test_loader), correct / total))
        acc = correct / total
        is_best = False
        if acc > best_test_acc:
            is_best = True
            best_test_acc = acc

        # 保存模型
        save_checkpoint({
            'epoch': epoch,
            'state_dict': [net.state_dict() for net in nets]
        }, directory, is_best)

        print("Epoch:{},test acc:{}(best:{})".format(epoch, acc, best_test_acc))

    start = 0
    has_joint_checkpoint = exist_checkpoint(train_jointly_directory)
    if not has_joint_checkpoint:
        # 这里需要专家网络首先单独训练30轮
        state = load_checkpoint_1(train_dependently_directory)
        if state is not None:
            start = state['epoch'] + 1
            for net, state_dict in zip(nets, state['state_dict']):
                net.load_state_dict(state_dict)
        for epoch in range(start, 30):
            train_with_clustering(epoch)
            val_with_clustering(epoch)
            test_with_clustering(epoch, train_dependently_directory)

    # 需要专家网络放在一起进行训练
    print("Start train jointly")
    if has_joint_checkpoint:
        state = load_checkpoint_1(train_jointly_directory)
        start = state['epoch'] + 1
        for net, state_dict in zip(nets, state['state_dict']):
            net.load_state_dict(state_dict)
    for epoch in range(start, n_epochs):
        train_jointly(epoch)
        val_with_clustering(epoch)
        test_with_clustering(epoch, train_jointly_directory)


if __name__ == '__main__':
    main()
