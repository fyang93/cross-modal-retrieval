import os
import time
import csv
import numpy as np
import joblib
import pickle
import torch
import torch.nn as nn
from PIL import Image


def load(path):
    """Load features
    """
    if not os.path.exists(path):
        raise Exception("{} does not exist".format(path))
    ext = os.path.splitext(path)[-1]
    if ext == '.pkl':
        with open(path, 'rb') as file:
            return pickle.load(file)
    return {'.npy': np, '.jbl': joblib}[ext].load(path)


def save(data, path):
    ext = os.path.splitext(path)[-1]
    if ext == '.npy':
        np.save(path, data)
    elif ext == '.jbl':
        joblib.dump(data, path)
    elif ext == '.pkl':
        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_images(image_paths, root_dir='', img_size=(144, 288)):
    images = []
    for path in image_paths:
        img = Image.open(os.path.join(root_dir, path))
        img = img.resize(img_size, Image.ANTIALIAS)
        images.append(np.array(img))
    return np.array(images)
