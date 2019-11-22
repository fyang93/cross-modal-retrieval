import os
import time
from datetime import datetime
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import SingleModalNet, CrossModalNet
import scipy.sparse as sparse
from dataset import SysuTestSet
from diffusion import Diffusion
from knn import KNN
from evaluate import eval_sysu
from sklearn import preprocessing

from IPython import embed


class Tester(object):
    def __init__(self, data_dir, cross_net, query_net, gallery_net, transform, mode='all', batch_size=128):
        self.query_net = query_net
        self.gallery_net = gallery_net
        self.cross_net = cross_net

        self.test_query_set = SysuTestSet(args.data_dir, 'thermal', transform, mode=args.mode)
        self.test_gallery_set = SysuTestSet(args.data_dir, 'visible', transform, mode=args.mode)

        if self.query_net is not None:
            self.query_net.eval()
            _, self.query_fc = self.query_net.extract(self.test_query_set, batch_size)
        if gallery_net is not None:
            self.gallery_net.eval()
            _, self.gallery_fc = self.gallery_net.extract(self.test_gallery_set, batch_size)
        cross_net.eval()
        _, self.cross_query_fc = self.cross_net.extract(self.test_query_set, batch_size)
        _, self.cross_gallery_fc = self.cross_net.extract(self.test_gallery_set, batch_size)


    def test(self):
        self.evaluate(- self.cross_query_fc @ self.cross_gallery_fc.T)

    def test_hetero(self, gamma=3):
        n_trunc = 1000
        kq, kd = 15, 30
        lam = 0.2

        diffusion = Diffusion(self.gallery_fc)
        inverse = diffusion.get_laplacian_inverse(n_trunc, kd)
        knn = KNN(self.cross_gallery_fc, method='cosine')
        sims, ids = knn.search(self.cross_query_fc, kq)
        sims[sims < 0] = 0
        sims /= np.sum(sims, axis=-1).reshape(-1, 1)
        sims = sims ** gamma
        scores_qg = np.empty((len(self.test_query_set), len(self.test_gallery_set)), dtype=np.float32)
        for i in range(len(self.test_query_set)):
            scores_qg[i] = (sims[i] @ inverse[ids[i]])

        diffusion = Diffusion(self.query_fc)
        inverse = diffusion.get_laplacian_inverse(n_trunc, kd)
        knn = KNN(self.cross_query_fc, method='cosine')
        sims, ids = knn.search(self.cross_gallery_fc, kq)
        sims[sims < 0] = 0
        sims /= np.sum(sims, axis=-1).reshape(-1, 1)
        sims = sims ** gamma
        scores_gq = np.empty((len(self.test_gallery_set), len(self.test_query_set)), dtype=np.float32)
        for i in range(len(self.test_gallery_set)):
            scores_gq[i] = (sims[i] @ inverse[ids[i]])

        scores = lam * scores_qg + (1 - lam) * scores_gq.T
        self.evaluate(-scores)

    def test_homo(self, gamma=3):
        n_trunc = 1000
        kq, kd = 15, 30
        lam = 0.2

        diffusion = Diffusion(self.cross_gallery_fc)
        inverse = diffusion.get_laplacian_inverse(n_trunc, kd)
        knn = KNN(self.cross_gallery_fc, method='cosine')
        sims, ids = knn.search(self.cross_query_fc, kq)
        sims[sims < 0] = 0
        sims /= np.sum(sims, axis=-1).reshape(-1, 1)
        sims = sims ** gamma
        scores_qg = np.empty((len(self.test_query_set), len(self.test_gallery_set)), dtype=np.float32)
        for i in range(len(self.test_query_set)):
            scores_qg[i] = (sims[i] @ inverse[ids[i]])

        diffusion = Diffusion(self.cross_query_fc)
        inverse = diffusion.get_laplacian_inverse(n_trunc, kd)
        knn = KNN(self.cross_query_fc, method='cosine')
        sims, ids = knn.search(self.cross_gallery_fc, kq)
        sims[sims < 0] = 0
        sims /= np.sum(sims, axis=-1).reshape(-1, 1)
        sims = sims ** gamma
        scores_gq = np.empty((len(self.test_gallery_set), len(self.test_query_set)), dtype=np.float32)
        for i in range(len(self.test_gallery_set)):
            scores_gq[i] = (sims[i] @ inverse[ids[i]])

        scores = lam * scores_qg + (1 - lam) * scores_gq.T
        self.evaluate(-scores)

    def evaluate(self, dists):
        for trial in range(10):
            self.test_gallery_set.random_sample()
            cmc_fc, mAP_fc = eval_sysu(dists[:, self.test_gallery_set.samples],
                                       self.test_query_set.labels,
                                       self.test_gallery_set.sampled_labels,
                                       self.test_query_set.cameras,
                                       self.test_gallery_set.sampled_cameras)

            print('Test Trial: {}'.format(trial))
            print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
                cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19]))
            print('mAP: {:.2%}'.format(mAP_fc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument('--arch', default='resnet50', type=str, help='Backbone architecture')
    parser.add_argument('--h', default=288, type=int, help='Image height')
    parser.add_argument('--w', default=144, type=int, help='Image width')
    parser.add_argument('--query_model_path', default='', type=str, help='Path to query-side model')
    parser.add_argument('--gallery_model_path', default='', type=str, help='Path to gallery-side model')
    parser.add_argument('--cross_model_path', default='', type=str, help='Path to cross-modality model')
    parser.add_argument('--hidden_dim', default=512, type=int, help='Hidden feature dimension of FC layer')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for evaluation')
    parser.add_argument('--mode', default='all', type=str, choices=['all', 'indoor'], help='Use indoor images or all images')
    parser.add_argument('--trial', default=1, type=int, help='RegDB only')
    args = parser.parse_args()
    args.feat_dim = {"resnet18": 512, "resnet50": 2048}[args.arch]
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_classes = 395 #sysu

    print('==> Building model..')
    if os.path.exists(args.query_model_path):
        query_net = SingleModalNet(args.hidden_dim, n_classes, arch=args.arch).to(device)
        print('==> loading checkpoint {}'.format(args.query_model_path))
        checkpoint = torch.load(args.query_model_path)
        query_net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'.format(args.query_model_path, checkpoint['epoch']))
    else:
        query_net = None

    if os.path.exists(args.gallery_model_path):
        gallery_net = SingleModalNet(args.hidden_dim, n_classes, arch=args.arch).to(device)
        print('==> loading checkpoint {}'.format(args.gallery_model_path))
        checkpoint = torch.load(args.gallery_model_path)
        gallery_net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'.format(args.gallery_model_path, checkpoint['epoch']))
    else:
        gallery_net = None

    cross_net = CrossModalNet(args.hidden_dim, n_classes, arch=args.arch).to(device)
    print('==> loading checkpoint {}'.format(args.cross_model_path))
    checkpoint = torch.load(args.cross_model_path)
    cross_net.load_state_dict(checkpoint['net'])
    print('==> loaded checkpoint {} (epoch {})'.format(args.cross_model_path, checkpoint['epoch']))

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.h, args.w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tester = Tester(args.data_dir, cross_net, query_net, gallery_net, transform_test, mode=args.mode)
    print('==> k-NN')
    tester.test()
    print('==> Proposed Heterogeneous')
    tester.test_hetero()
    print('==> Proposed Homogeneous')
    tester.test_homo()
