import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SingleModalNet(nn.Module):
    def __init__(self, hidden_dim, n_classes, drop=0.5, arch='resnet50'):
        super().__init__()
        self.feat_dim = {"resnet18": 512, "resnet50": 2048}[arch]
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.backbone = Backbone(arch)
        self.feature = FeatureBlock(self.feat_dim, self.hidden_dim)
        self.classifier = ClassBlock(self.hidden_dim, self.n_classes, dropout=drop)
        self.l2norm = lambda x: F.normalize(x, p=2, dim=1)

    def forward(self, x):
        pool = self.backbone(x)
        fc = self.feature(pool)
        if self.training:
            out = self.classifier(fc)
            return self.l2norm(fc), out
        else:
            return self.l2norm(pool), self.l2norm(fc)

    def extract(self, dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8)
        pool = np.zeros((len(dataset), self.feat_dim))
        fc = np.zeros((len(dataset), self.hidden_dim))
        with torch.no_grad():
            for i, images in enumerate(loader):
                images = images.cuda()
                pool_batch, fc_batch = self.forward(images)
                pool[i * batch_size:(i+1) *
                     batch_size] = pool_batch.detach().cpu().numpy()
                fc[i * batch_size:(i+1) *
                   batch_size] = fc_batch.detach().cpu().numpy()
        return pool, fc


class CrossModalNet(nn.Module):
    def __init__(self, hidden_dim, n_classes, drop=0.0, arch='resnet50'):
        super().__init__()
        self.feat_dim = {"resnet18": 512, "resnet50": 2048}[arch]
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.a_net = Backbone(arch)
        self.b_net = Backbone(arch)
        self.feature = FeatureBlock(self.feat_dim, self.hidden_dim)
        self.classifier = ClassBlock(self.hidden_dim, self.n_classes, dropout=drop)
        self.l2norm = lambda x: F.normalize(x, p=2, dim=1)

    def forward(self, x, branch):
        net = {'a': self.a_net, 'b': self.b_net}[branch]
        pool = net(x)
        fc = self.feature(pool)
        if self.training:
            out = self.classifier(fc)
            return self.l2norm(fc), out
        else:
            return self.l2norm(pool), self.l2norm(fc)

    def extract(self, dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8)
        pool = np.zeros((len(dataset), self.feat_dim))
        fc = np.zeros((len(dataset), self.hidden_dim))
        with torch.no_grad():
            for i, images in enumerate(loader):
                images = images.cuda()
                pool_batch, fc_batch = self.forward(images, dataset.branch)
                pool[i * batch_size:(i+1) *
                     batch_size] = pool_batch.detach().cpu().numpy()
                fc[i * batch_size:(i+1) *
                   batch_size] = fc_batch.detach().cpu().numpy()
        return pool, fc


class Backbone(nn.Module):
    def __init__(self, arch):
        super().__init__()
        backbone = getattr(models, arch)(pretrained=True)
        if arch.startswith('resnet'):
            backbone = list(backbone.children())[:-2]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x, p=3, eps=1e-6):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)


class FeatureBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, hidden_dim)]
        feat_block += [nn.BatchNorm1d(hidden_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, n_classes, dropout=0.0):
        super().__init__()
        class_block = []
        class_block += [nn.LeakyReLU(0.1)]
        class_block += [nn.Dropout(p=dropout)]
        class_block += [nn.Linear(input_dim, n_classes)]
        class_block = nn.Sequential(*class_block)
        class_block.apply(weights_init_classifier)

        self.class_block = class_block

    def forward(self, x):
        x = self.class_block(x)
        return x
