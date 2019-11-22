import os
from datetime import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import SingleModalNet
from dataset import SysuTrainSet, SysuTestSet
from evaluate import eval_sysu
from tqdm import tqdm
import itertools

from IPython import embed


def train(epoch):
    adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr)

    net.train()
    with tqdm(total=len(train_loader), desc='[{}]'.format(epoch)) as t:
        for i, (images, labels) in enumerate(train_loader):
            i_cur = epoch * len(train_loader) + i

            images = images.to(device)
            labels = labels.to(device)

            _, out = net(images)
            loss = cross_entropy(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.update()


def test(epoch):
    net.eval()
    _, fc = net.extract(test_set, args.batch_test)

    # compute the distances (negative similarities)
    dists_fc = - fc @ fc.T
    np.fill_diagonal(dists_fc, 1)

    # evaluation
    if args.dataset == 'sysu':
        cmc, mAP = eval_sysu(dists_fc, test_set.labels, test_set.labels, test_set.cameras, test_set.cameras)
    elif args.dataset == 'regdb':
        cmc, mAP = eval_regdb(dists_fc, test_set.labels, test_set.labels)
    return cmc, mAP


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_base):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = lr_base * decay
    for grp in optimizer.param_groups:
        grp['lr'] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset files")
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet50', help='Backbone architecture')
    parser.add_argument('--modal', type=str, required=True, help='Modality')
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--lr_steps', type=float, default=[20, 30], nargs="+",
                        help='Epochs to decay learning rate by dividing 10')
    parser.add_argument('--resume', '-r', type=str, default='',
                        help='Path to checkpoint you want to resume from')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden feature dimension of FC layer')
    parser.add_argument('--batch_train', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--batch_test', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--h', type=int, default=288, help='Image height')
    parser.add_argument('--w', type=int, default=144, help='Image width')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout ratio')
    parser.add_argument('--margin', default=0.5, type=float, help='Triplet margin')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'indoor'],
                        help='Use indoor images or all images')
    parser.add_argument('--trial', type=int, default=1, help='RegDB only')
    args = parser.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    args.feat_dim = {"resnet18": 512, "resnet50": 2048}[args.arch]
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_acc = 0  # best test Rank-1
    start_epoch = 0

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((args.h, args.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.h, args.w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = SysuTrainSet(args.data_dir, args.modal, transform_train)
    test_set = SysuTestSet(args.data_dir, args.modal, transform_test, mode=args.mode)
    train_loader = DataLoader(train_set, batch_size=args.batch_train, shuffle=True, num_workers=8)

    print('==> Building model..')
    net = SingleModalNet(args.hidden_dim, train_set.n_classes, drop=args.drop, arch=args.arch).to(device)

    base_params = {'params': net.backbone.parameters(), 'lr': 0.1 * args.lr}
    custom_params = {'params': itertools.chain(net.feature.parameters(), net.classifier.parameters()), 'lr': args.lr}
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD([base_params, custom_params], weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam([base_params, custom_params], weight_decay=5e-4)

    if args.resume and os.path.isfile(args.resume):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'.format(
            args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

    cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    desc = f'{args.dataset}_{args.modal}_{args.arch}_drop_{args.drop}_lr_{args.lr}_hid_{args.hidden_dim}_{args.optim}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    print('==> Start Training...')
    for epoch in range(start_epoch, 30):
        train(epoch)

        if (epoch + 1) % 2 == 0:
            print('Test Epoch: {}'.format(epoch))
            cmc_fc, mAP_fc = test(epoch)
            print(f'==> Rank-1: {cmc_fc[0]:.2%} | Rank-5: {cmc_fc[4]:.2%} | Rank-10: {cmc_fc[9]:.2%}| mAP: {mAP_fc:.2%}')

            # save best model
            if cmc_fc[0] > best_acc:
                best_acc = cmc_fc[0]
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc_fc,
                    'mAP': mAP_fc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(args.ckpt_dir, f'{desc}.pth'))
