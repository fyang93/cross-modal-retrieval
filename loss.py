import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistMat(nn.Module):
    def __init__(self, metric='euclidean', norm=False, min=1e-12, max=1e+12):
        super().__init__()
        assert metric in ['euclidean', 'sqeuclidean', 'cosine']
        self.metric = metric
        self.l2norm = lambda x: F.normalize(x, p=2, dim=1)
        self.norm = norm
        self.min = min
        self.max = max

    def forward(self, x, y=None):
        """
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            y: optional tensor with shape (?, feat_dim)
        Return:
            distmat with shape (batch_size, batch_size) when y is None,
            distmat with shape (batch_size, ?) otherwise
        """
        if self.norm:
            x = self.l2norm(x)
            y = self.l2norm(y) if y is not None else None

        if y is None:
            n = x.size(0)
            if self.metric.endswith('euclidean'):
                distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
                distmat = distmat + distmat.t()
                distmat.addmm_(1, -2, x, x.t())
            elif self.metric == 'cosine':
                distmat = x @ x.t()
        else:
            assert x.size(1) == y.size(1)
            if self.metric.endswith('euclidean'):
                n, m = x.size(0), y.size(0)
                distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, m) + \
                          torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
                distmat.addmm_(1, -2, x, y.t())
            elif self.metric == 'cosine':
                distmat = x @ y.t()

        distmat = distmat.clamp(min=self.min, max=self.max)
        if self.metric == 'euclidean':
            distmat = distmat.sqrt()
        return distmat


class TripletLoss(nn.Module):
    """ref: https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py"""

    def __init__(self, margin=0.3, metric='euclidean'):
        super().__init__()
        self.margin = margin
        self.metric = metric
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = DistMat(metric=self.metric)(x)
        mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())

        if self.metric.endswith('euclidean'):
            dists_ap = torch.stack([distmat[i][mask[i]].max() for i in range(batch_size)])
            dists_an = torch.stack([distmat[i][mask[i] == 0].min() for i in range(batch_size)])
            y = torch.ones_like(dists_an)
            loss = self.ranking_loss(dists_an, dists_ap, y)
        elif self.metric == 'cosine':
            dists_ap = torch.stack([distmat[i][mask[i]].min() for i in range(batch_size)])
            dists_an = torch.stack([distmat[i][mask[i] == 0].max() for i in range(batch_size)])
            y = torch.ones_like(dists_an)
            loss = self.ranking_loss(dists_ap, dists_an, y)

        return loss
