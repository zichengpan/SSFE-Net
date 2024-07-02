import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
import numpy as np
from torchvision import models
from torchvision import transforms

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class MYNET(nn.Module):

    def __init__(self, args, mode=None, strategy='SSFE'):
        super().__init__()

        self.mode = mode
        self.args = args

        if self.args.dataset in ['mini_imagenet', 'PlantVillage']:
            if strategy == 'SSL':
                self.encoder = models.resnet50(pretrained=False)
                self.encoder.fc = Identity()
                self.num_features = 2048
                self.kl_loss = nn.KLDivLoss()

            else:
                self.encoder = resnet18(False, args)  # pretrained=False
                self.num_features = 512
        if self.args.dataset == 'cub200':
            if strategy == 'SSL':
                self.encoder = models.resnet50(pretrained=False)
                self.encoder.fc = Identity()
                self.num_features = 2048
                self.kl_loss = nn.KLDivLoss()

            else:
                self.encoder = resnet18(True, args)  # pretrained=False
                self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if strategy == 'SSL':
            self.feature_extractor = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512, affine=False),  # Page:5, Paragraph:2
                nn.ReLU(inplace=True),
            )
            self.predictor = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512)
            )
            self.converter = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)
            )

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)


    def forward_metric(self, x):
        feat = self.encode(x)
        x = F.linear(F.normalize(feat, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
        x = self.args.temperature * x

        return x, feat


    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)


    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def ssl_train_loop(self, epoch, train_loader, optimizer):
        self.train()

        avg_loss = 0
        for i, batch in enumerate(train_loader):

            data, train_label = [_.cuda() for _ in batch]

            x = data.reshape([data.shape[0] * data.shape[1], *data.shape[2:]])
            loss = self.contrastive_loss(x)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

        print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))

    def contrastive_loss(self, x):
        x1 = x.clone()
        x2 = x.clone()
        for index in range(x.shape[0]):

            x1[index] = self.data_augmentation(x[index])
            x2[index] = self.data_augmentation(x[index])

        z1 = self.f(x1)
        z2 = self.f(x2)
        p1, p2 = self.h(z1), self.h(z2)
        loss = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        return loss

    def data_augmentation(self, img):

        x = transforms.RandomHorizontalFlip()(img)
        if np.random.random() < 0.8:
            x = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(x)
        else:
            x = transforms.RandomGrayscale(p=1.0)(x)
        x = transforms.GaussianBlur((5, 5))(x)
        return x

    def ssl_test_loop(self, test_loader, args):
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            x = x.cuda()
            args.SSL_query = x.size(1) - args.SSL_shot  # x:[N, S+Q, n_channel, h, w]
            args.SSL_way = x.size(0)

            with torch.no_grad():
                x = x.reshape(args.SSL_way * (args.SSL_shot + args.SSL_query), *x.size()[2:])
                z_all = self.encoder.forward(x)
                z_all = z_all.reshape(args.SSL_way, args.SSL_shot + args.SSL_query, *z_all.shape[1:])  # [N, S+Q, d]
                z_support = z_all[:, :args.SSL_shot]  # [N, S, d]
                z_query = z_all[:, args.SSL_shot:]  # [N, Q, d]
                z_proto = z_support.reshape(args.SSL_way, args.SSL_shot, -1).mean(1)  # [N,d]
                z_query = z_query.reshape(args.SSL_way * args.SSL_query, -1)  # [N*Q,d]
                scores = self.cosine_similarity(z_query, z_proto)
                y_query = np.repeat(range(args.SSL_way), args.SSL_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
                topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
                topk_ind = topk_labels.cpu().numpy()  # index of topk
                acc_all.append(np.sum(topk_ind[:, 0] == y_query) / len(y_query) * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('Test Acc = %4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean


    def cosine_similarity(self, x, y):

        assert x.size(1) == y.size(1)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))
        y = y.unsqueeze(0).expand(x.shape)
        return (x * y).sum(2)

    def f(self, x):
        x = self.encoder(x)
        x = self.converter(x)
        x = self.feature_extractor(x)
        return x

    def h(self, x):
        x = self.predictor(x)
        return x

    def D(self, p, z):
        z = z.detach()
        p = torch.nn.functional.normalize(p, dim=1)
        z = torch.nn.functional.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()