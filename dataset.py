import os
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from preprocess import get_image_paths
import utils


class CrossTrainSet(Dataset):
    def __init__(self, images_a, labels_a, images_b, labels_b, transform, batch_size):
        self.images_a = images_a
        self.labels_a = labels_a
        self.images_b = images_b
        self.labels_b = labels_b
        # unique(labels_a) == unique(labels_b)
        self._uni_labels = np.unique(labels_a)
        self._label_dict_a = defaultdict(list)
        self._label_dict_b = defaultdict(list)
        for index, label in enumerate(self.labels_a):
            self._label_dict_a[label].append(index)
        for index, label in enumerate(self.labels_b):
            self._label_dict_b[label].append(index)
        self.N = max(len(self.labels_a), len(self.labels_b))
        self.n_classes = len(self._uni_labels)
        self.batch_size = batch_size
        self.transform = transform
        self.reset()

    def reset(self):
        for i_batch in range(self.N // self.batch_size):
            # randomly choose #batch_size labels
            batch_labels = np.random.choice(self._uni_labels, self.batch_size, replace=False)
            # randomly sample one image for each person
            samples_a = [np.random.choice(self._label_dict_a[l]) for l in batch_labels]
            samples_b = [np.random.choice(self._label_dict_b[l]) for l in batch_labels]
            if i_batch == 0:
                self.aids = samples_a
                self.bids = samples_b
            else:
                self.aids = np.hstack((self.aids, samples_a))
                self.bids = np.hstack((self.bids, samples_b))

    def __getitem__(self, index):
        try:
            aid, bid = self.aids[index], self.bids[index]
        except IndexError:
            self.reset()
            raise IndexError

        image_a = self.transform(self.images_a[aid])
        image_b = self.transform(self.images_b[bid])
        label_a = self.labels_a[aid]
        label_b = self.labels_b[bid]
        return image_a, image_b, label_a, label_b

    def __len__(self):
        return len(self.aids)


class TrainSet(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.n_classes = len(np.unique(self.labels))

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)


class TestSet(Dataset):
    def __init__(self, image_paths, labels, transform, root_dir=''):
        self.image_paths = image_paths
        self.images = utils.load_images(image_paths, root_dir)
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.labels)


class SysuCrossTrainSet(CrossTrainSet):
    def __init__(self, data_dir, transform, batch_size):
        visible_images = utils.load(os.path.join(data_dir, 'train_visible_images.npy'))
        visible_labels = utils.load(os.path.join(data_dir, 'train_visible_labels.npy'))
        thermal_images = utils.load(os.path.join(data_dir, 'train_thermal_images.npy'))
        thermal_labels = utils.load(os.path.join(data_dir, 'train_thermal_labels.npy'))
        super().__init__(visible_images, visible_labels, thermal_images, thermal_labels, transform, batch_size)


class SysuTrainSet(TrainSet):
    def __init__(self, data_dir, modal, transform):
        assert modal in ['visible', 'thermal']
        images = utils.load(os.path.join(data_dir, f'train_{modal}_images.npy'))
        labels = utils.load(os.path.join(data_dir, f'train_{modal}_labels.npy'))
        super().__init__(images, labels, transform)


class SysuTestSet(TestSet):
    def __init__(self, data_dir, modal, transform, mode='all'):
        assert modal in ['visible', 'thermal']
        self.data_dir = data_dir
        self.modal = modal
        self.branch = {'visible': 'a', 'thermal': 'b'}[modal]
        self.mode = mode
        image_paths, _, _ = get_image_paths(data_dir, ['test'], modal, mode)
        labels = np.array([int(p[-13:-9]) for p in image_paths])
        self.cameras = np.array([int(p[-15]) for p in image_paths])
        super().__init__(image_paths, labels, transform)

    def random_sample(self):
        _, _, self.samples = get_image_paths(self.data_dir, ['test'], self.modal, self.mode)
        self.sampled_cameras = np.array([int(self.image_paths[i][-15]) for i in self.samples])
        self.sampled_labels = np.array([int(self.image_paths[i][-13:-9]) for i in self.samples])
