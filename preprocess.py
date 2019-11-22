" SYSU pre-process "

import os
import argparse
import numpy as np
from PIL import Image
from glob import glob
import utils


def get_image_paths(data_dir, subsets, modal, mode='all'):
    """Get person ids and image paths for given subsets and modality
    Args:
        data_dir: directory to SYSU dataset
        subsets: a list whose elements can be 'train', 'test' and 'val'
        modal: modal of data, 'ir' or 'rgb'
        mode: all images or indoor images only
    """
    assert all([s in ['train', 'test', 'val'] for s in subsets])
    assert modal in ['visible', 'thermal']
    assert mode in ['all', 'indoor']

    if modal == 'visible':
        cameras = {'all': ['cam1', 'cam2', 'cam4', 'cam5'],
                   'indoor': ['cam1', 'cam2']}[mode]
    if modal == 'thermal':
        cameras = ['cam3', 'cam6']

    person_ids = []
    for subset in subsets:
        with open(os.path.join(data_dir, 'exp/{}_id.txt'.format(subset)), 'r') as f:
            person_ids.extend([int(i) for i in f.readlines()[0].split(',')])
    person_ids = sorted(person_ids)

    samples = []
    image_paths = []

    for i in person_ids:
        for cam in cameras:
            img_pattern = os.path.join(data_dir, cam, '{:0>4d}/*.jpg'.format(i))
            paths = sorted(glob(img_pattern))
            if any(paths):
                samples.append(len(image_paths) + np.random.choice(len(paths)))
                image_paths.extend(paths)

    return image_paths, person_ids, samples


def gen_pairs(image_paths, person_ids, img_w=144, img_h=288):
    pid2label = {pid: label for label, pid in enumerate(person_ids)}
    images, labels = [], []
    for img_path in image_paths:
        img = Image.open(img_path)
        img = img.resize((img_w, img_h), Image.ANTIALIAS)
        images.append(np.array(img))
        pid = int(img_path[-13:-9])
        labels.append(pid2label[pid])
    return np.array(images), np.array(labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to SYSU-MM01 dataset")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    for subset in ['train', 'test']:
        for modal in ['visible', 'thermal']:
            subsets = {'train': ['train', 'val'], 'test': ['test']}[subset]
            image_paths, pids, _ = get_image_paths(args.data_dir, subsets, modal)
            images, labels = gen_pairs(image_paths, pids)
            utils.save(images, os.path.join(args.data_dir, f"{subset}_{modal}_images.npy"))
            utils.save(labels, os.path.join(args.data_dir, f"{subset}_{modal}_labels.npy"))
