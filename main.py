import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from dataset import Cub2011, Cub2011Cluster
from net import *
# from utils import *
from utils import setup_seed, load_checkpoint_1, save_checkpoint, adjust_learning_rate, exist_checkpoint

best_test_acc = 0
best_val_acc = 0


def main():
    setup_seed(2)

    # show_result(use_lda=False)
    # nets = [ClassificationAlexNet(n_labels) for _ in range(n_clusters)]

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

    # datasets = [Cub2011Cluster(root=dataset_directory, cluster_id=i, train=True, transform=train_transform)
    #             for i in range(n_clusters)]
    # # datasets = [Cub2011(root=dataset_directory  train=True, transform=transform)
    # #             for i in range(n_clusters)]
    # datasets_len = [len(dataset) for dataset in datasets]
    # data_loaders = [DataLoader(dataset, shuffle=True, batch_size=batch_size) for dataset in datasets]

    train_set = Cub2011(root=dataset_directory, train=True, transform=train_transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    test_set = Cub2011Cluster(root=dataset_directory, train=False, transform=test_transform)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    val_set = Cub2011(root=dataset_directory, train=True, transform=test_transform)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size)

    cluster_net = ClusterAlexNet()

    nets = [ClassificationAlexNet(n_labels) for i in range(n_clusters)]
    # nets = [nn.DataParallel(net) for net in nets]
    if torch.cuda.is_available():
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
                    if torch.cuda.is_available():
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
        optimizer = torch.optim.SGD([{'params': net.parameters()} for net in nets], lr=lr, momentum=0.9,
                                    weight_decay=1e-3)
        # adjust_learning_rate(optimizer, epoch)
        correct, total, train_loss = 0, 0, 0
        for batch_id, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            # c_s = []
            # z_s = []
            # for net in nets:
            #     z = net(x)
            #     z_s.append(z)
                # c_s.append(z.max(dim=1).values)
            # (n,6)
            # c = torch.stack(c_s, dim=1)
            # (n,6,1)
            # a = F.softmax(c, dim=1).unsqueeze(-1)

            # z1 = torch.stack(z_s, dim=1)
            # (n, 200)
            # z = (a * z1).sum(1)
            # z = z1.sum(1)
            z = nets[0](x)
            loss = criterion(z, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # (n)
            y_ = z.argmax(1)
            correct += (y_ == y).sum().cpu().item()
            total += y.shape[0]
            print("Epoch:{},train batch:{}/{}, loss:{} ,acc:{}".format(epoch, batch_id, len(train_loader),
                                                                       train_loss / (batch_id + 1), correct / total))

    def val_with_clustering(epoch):
        global best_val_acc
        correct, total = 0, 0
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(val_loader):
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                # c_s = []
                z_s = []
                for net in nets:
                    z = net(x)
                    z_s.append(z)
                    # c_s.append(z.max(dim=1).values)
                # (n,6)
                # c = torch.stack(c_s, dim=1)
                # (n,6,1)
                # a = F.softmax(c, dim=1).unsqueeze(-1)

                z1 = torch.stack(z_s, dim=1)
                # z = (a * z1).sum(1)
                z = z1.sum(1)
                y_ = z.argmax(1)
                correct += (y_ == y).sum().cpu().item()
                total += y.shape[0]
                print("Epoch:{}, val batch:{}/{},acc:{}".format(epoch, batch_id, len(val_loader), correct / total))
        acc = correct / total
        best_val_acc = max(best_val_acc, acc)
        print("Epoch:{},val acc:{}(best:{})".format(epoch, acc, best_val_acc))

    def test_with_clustering(epoch, directory):
        global best_test_acc
        correct, total = 0, 0
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(test_loader):
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                # c_s = []
                z_s = []
                # for net in nets:
                #     z = net(x)
                #     z_s.append(z)
                    # c_s.append(z.max(dim=1).values)
                # (n,6)
                # c = torch.stack(c_s, dim=1)
                # (n,6,1)
                # a = F.softmax(c, dim=1).unsqueeze(-1)

                # z1 = torch.stack(z_s, dim=1)
                # z = (a * z1).sum(1)
                # z = z1.sum(1)
                z = nets[0](x)
                y_ = z.argmax(1)
                correct += (y_ == y).sum().cpu().item()
                total += y.shape[0]
                print("Epoch:{},test batch:{}/{},acc:{}".format(epoch, batch_id, len(test_loader), correct / total))
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

    # def test(epoch):
    #     with open('lda.pickle', 'rb') as f:
    #         lda = pickle.load(f)
    #     with open('kmean.pickle', 'rb') as f:
    #         kmean = pickle.load(f)
    #     for batch_id, (x, y) in enumerate(test_loader):
    #         x, y = x.cuda(), y.cuda()
    #         features = cluster_net(x)
    #         features = lda.transform(features)
    #         cluster_ids = kmean.fit_predict(features)
    #         print(cluster_ids)



    start = 0
    has_joint_checkpoint = exist_checkpoint(train_jointly_directory)
    # if not has_joint_checkpoint:
    #     print("Start train dependently")
    #     # 这里需要专家网络首先单独训练30轮
    #     state = load_checkpoint_1(train_dependently_directory)
    #     if state is not None:
    #         start = state['epoch'] + 1
    #         for net, state_dict in zip(nets, state['state_dict']):
    #             net.load_state_dict(state_dict)
    #     for epoch in range(start, 30):
    #         train_with_clustering(epoch)
    #         # val_with_clustering(epoch)
    #         test_with_clustering(epoch, train_dependently_directory)
    # 需要专家网络放在一起进行训练
    print("Start train jointly")
    if has_joint_checkpoint:
        state = load_checkpoint_1(train_jointly_directory)
        start = state['epoch'] + 1
        for net, state_dict in zip(nets, state['state_dict']):
            net.load_state_dict(state_dict)
    for epoch in range(start, n_epochs):
        train_jointly(epoch)
        # val_with_clustering(epoch)
        test_with_clustering(epoch, train_jointly_directory)


if __name__ == '__main__':
    main()
