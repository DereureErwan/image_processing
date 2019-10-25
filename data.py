"""
Implementing Data generator for U-Net feating
"""
import numpy as np
from skimage import io
import elasticdeform


class Data:
    """
    Class for data augmentation
    """
    TRAIN_DATA = 'data/train-volume.tif'
    TRAIN_LABELS = 'data/train-labels.tif'
    TEST_DATA = 'data/test-volume.tif'

    def __init__(self, sigma=25, points=3, shuffle=False):
        self.sigma = sigma
        self.points = points
        self.shuffle = shuffle
        self._load_images()

    def _load_images(self):
        self.train = io.imread(self.TRAIN_DATA) / 255
        self.n_samples = self.train.shape[0]
        self.labels = io.imread(self.TRAIN_LABELS) / 255
        self.test = io.imread(self.TEST_DATA) / 255
        self.size = self.train.shape[1]

    def train_generator(self, batch_size):
        # TODO: create tiles and their associated labels
        idx = np.arange(0, self.n_samples)
        if self.shuffle:
            idx = np.random.shuffle(idx)
        for i in range(0, self.n_samples, batch_size):
            # label = self.labels[idx[i: i + batch_size]]
            img = np.zeros((batch_size, self.size, self.size))
            label = np.zeros((batch_size, self.size, self.size))
            for j in range(i, i+batch_size):
                img[j-i], label[j-1] =\
                    elasticdeform.deform_random_grid(
                        [self.train[idx[j]], self.labels[idx[j]]],
                        sigma=self.sigma, points=self.points
                    )
            label = np.where(label <= 0.5, 0, 1)
            yield img, label

    def test_generator(self, batch_size):
        for i in range(0, self.test.shape[0], batch_size):
            yield self.test[i: i + batch_size, :, :]
