import os

import numpy as np
from numpy.core import asarray
from skimage import io
from sklearn.cluster import DBSCAN
from tifffile import imwrite
from tqdm import tqdm


class LossWeights:
    LABELS_PATH = 'data/train-labels.tif'
    WEIGHT_PATH = 'data/train-labels-weights.tif'

    def __init__(self, w0=10, sigma=5, compute=False):
        self.w0 = w0
        self.sigma = sigma
        self.compute = compute

    def __call__(self):
        labels = io.imread(self.LABELS_PATH) / 255
        weights = []
        if not os.path.isfile(self.WEIGHT_PATH) or self.compute:
            for y in tqdm(labels, desc='Computing the weigths map for each label'):
                weights.append(self._compute_weights(y).reshape(1, 512, 512))
            weights = np.vstack(weights)
            # XXX: check args for imwrite to save a readable image
            imwrite(self.WEIGHT_PATH, weights)
        else:
            weights = io.imread(self.WEIGHT_PATH)
        self._weights = weights
        return self._weights.copy()

    def __len__(self):
        return self.weights.shape[0]

    def __getitem__(self, key):
        return self.weights.__getitem__(key)

    def __setitem__(self, index, value):
        self.weights[index] = asarray(value, self.weights.dtype)

    def __delitem__(self, key):
        return self.weights.__delitem__(key)

    @property
    def shape(self):
        return self.weights.shape

    @property
    def weights(self):
        if not hasattr(self, '_weights'):
            _ = self()
        return self._weights.copy()

    def _compute_weights(self, y):
        """
        Compute weight matrix as indicated in the U-Net paper
        for an image label
        """
        d1, d2 = self._compute_distances(y)
        return (
            np.mean(y) * y
            + self.w0 * np.exp(-(
                (d1 + d2)**2) / (2 * self.sigma**2)
            ))

    def _compute_clusters(self, y):
        """
        Compute clusters when there is only
        a binary choice (0 and 1) in order to compute d2
        """
        y = y.copy()
        idx = np.nonzero(y)
        X = DBSCAN(eps=np.sqrt(2)).fit_predict(np.transpose(idx))
        y[idx] = X + 1  # we add one because labels start at 0
        return y

    def _compute_distances(self, y):
        """
        Compute d1 and d2 distances
        d1 : distance to nearest cell
        d2 : distance to second nearest cell
        """
        # XXX: this method needs to be optimized, its very slow for now
        y = self._compute_clusters(y)

        d1 = np.zeros(y.shape)
        d2 = np.zeros(y.shape)

        # d1 and d2 distance for black pixel (not in a cell)
        rows, cols = np.where(y == 0)
        for i, j in zip(rows, cols):
            dist = 1
            dist1 = None
            dist2 = None
            while True:
                square = set(y[
                    max(i-dist, 0): min(i+dist+1, y.shape[0]),
                    max(j-dist, 0): min(j+dist+1, y.shape[1])
                ].flatten())
                square.remove(0)
                if square:
                    if len(square) > 1:
                        dist1 = dist2 = dist
                        break
                    if dist1 is None:
                        dist1 = dist
                        dist += 1
                        continue
                    dist2 = dist
                    break
                dist += 1
            d1[i, j] = dist1
            d2[i, j] = dist2

        # d2 distance for pixel that are in a cell
        rows, cols = np.where(y != 0)
        for i, j in zip(rows, cols):
            dist = 1
            label = y[i, j]
            while True:
                square = set(y[
                    max(i-dist, 0): min(i+dist+1, y.shape[0]),
                    max(j-dist, 0): min(j+dist+1, y.shape[1])
                ].flatten())
                square.remove(label)
                try:
                    square.remove(0)
                except KeyError:
                    pass
                if square:
                    break
                dist += 1
            d2[i, j] = dist

        return d1, d2
