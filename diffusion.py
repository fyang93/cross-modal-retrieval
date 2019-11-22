#!/usr/bin/env python
# -*- coding: utf-8 -*-

" diffusion module "

import os
import time
import numpy as np
import joblib
from joblib import Parallel, delayed
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from tqdm import tqdm
from knn import KNN, ANN


trunc_ids = None
trunc_init = None
laplacian = None


def get_column_vector(i):
    ids = trunc_ids[i]
    trunc_lap = laplacian[ids][:, ids]
    scores, _ = linalg.cg(trunc_lap, trunc_init, tol=1e-6, maxiter=20)
    return scores


class Diffusion(object):
    """Diffusion class
    """
    def __init__(self, features):
        self.features = features
        self.N = len(self.features)
        self.knn = KNN(self.features, method='cosine')

    def get_laplacian_inverse(self, n_trunc, kd, gamma=3):
        """Get pseudo inverse of Laplacian
        """
        global trunc_ids, trunc_init, laplacian
        sims, ids = self.knn.search(self.features, n_trunc)
        trunc_ids = ids
        laplacian = self.get_laplacian(sims[:, :kd], ids[:, :kd], gamma)
        trunc_init = np.zeros(n_trunc)
        trunc_init[0] = 1

        results = Parallel(n_jobs=-1, prefer='threads')(delayed(get_column_vector)(i)
                                      for i in tqdm(range(self.N),
                                                    desc='[diffusion]'))

        rows = np.repeat(np.arange(self.N), n_trunc)
        inverse = sparse.csr_matrix((np.concatenate(results), (rows, trunc_ids.reshape(-1))),
                                    shape=(self.N, self.N), dtype=np.float32)
        return inverse

    def get_laplacian(self, sims, ids, gamma=3, alpha=0.99):
        """Create Laplacian matrix for the mutual kNN graph of the whole dataset
        Args:
            sims: similarities of kNN
            ids: indexes of kNN
        Returns:
            laplacian: Laplacian matrix
        """
        num = sims.shape[0]
        sims[sims < 0] = 0  # similarity should be non-negative
        sims = sims ** gamma
        # vec_ids: feature vectors' ids
        # mut_ids: mutual (reciprocal) nearest neighbors' ids
        # mut_sims: similarites between feature vectors and their mutual nearest neighbors
        vec_ids, mut_ids, mut_sims = [], [], []
        for i in range(num):
            # check reciprocity: i is in j's kNN and j is in i's kNN
            ismutual = np.isin(ids[ids[i]], i).any(axis=1)
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
                mut_ids.append(ids[i, ismutual])
                mut_sims.append(sims[i, ismutual])
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix((mut_sims, (vec_ids, mut_ids)),
                                     shape=(num, num), dtype=np.float32)
        affinity[range(num), range(num)] = 0
        degrees = affinity @ np.ones(num) + 1e-12
        # mat: degree matrix ^ (-1/2)
        mat = sparse.dia_matrix(
            (degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32)
        transition = mat @ affinity @ mat
        sparse_eye = sparse.dia_matrix(
            (np.ones(num), [0]), shape=(num, num), dtype=np.float32)
        laplacian = sparse_eye - alpha * transition
        return laplacian
