"""
Implementing Data generator for U-Net feating
"""
import numpy as np
from scipy import ndimage
from skimage import io
import elasticdeform

from loss_weights import LossWeights


class Data:
    """
    Class for data augmentation
    """
    TRAIN_DATA = 'data/train-volume.tif'
    TRAIN_LABELS = 'data/train-labels.tif'
    TEST_DATA = 'data/test-volume.tif'

    def __init__(
        self,
        custom_loss=True,
        shuffle=False,
        sigma=25,
        points=3,
        force_zero=True,
        rotate=True,
        tile=267,
        label_shape=68,
        overlap=37,
        extension=199,
        iteration=12
    ):
        self.custom_loss = custom_loss
        self.shuffle = shuffle
        self._load_images()
        self.sigma = sigma
        self.points = points
        self.force_zero = force_zero
        self.rotate = rotate
        # TODO: compute those numbers automatically
        # for now they were obtained by hand
        self.tile = tile
        self.overlap = overlap
        self.label_shape = label_shape
        self.extension = extension
        self.iteration = iteration

    def _load_images(self):
        self.train = io.imread(self.TRAIN_DATA) / 255
        self.n_samples = self.train.shape[0]
        self.labels = io.imread(self.TRAIN_LABELS) / 255
        self.test = io.imread(self.TEST_DATA) / 255
        self.size = self.train.shape[1]
        if self.custom_loss:
            self.weights = LossWeights()()

    def _data_augmentation(self, X, y):
        if (self.sigma is not None) and (self.points is not None):
            X, y = elasticdeform.deform_random_grid(
                [X, y],
                sigma=self.sigma, points=self.points
            )
        if self.rotate:
            rotation = np.random.choice(np.linspace(0, 360, 10))
            X, y = ndimage.rotate(X, rotation), ndimage.rotate(y, rotation)
        if self.force_zero:
            y = np.where(y <= 0.5, 0, 1)
        return X, y

    def generator(self, train=True):
        # TODO: Accept batch size of more than one
        while True:
            idx = np.arange(0, self.n_samples)
            if self.shuffle:
                idx = np.random.shuffle(idx)
            for s in idx:
                img = self.train[s]
                if train:
                    label = self.labels[s]
                    img, label = self._data_augmentation(img, label)
                    if self.custom_loss:
                        weights = self.weights[s]
                img = np.pad(img, self.extension, mode='reflect')
                for i in range(self.iteration):
                    for j in range(self.iteration):
                        img_ = img[
                            i*self.overlap: self.tile + i*self.overlap,
                            j*self.overlap: self.tile + j*self.overlap
                        ].reshape((1,)+(self.tile, self.tile)+(1,))
                        if train:
                            label_ = label[
                                i*self.overlap: self.label_shape + i*self.overlap,
                                j*self.overlap: self.label_shape + j*self.overlap
                            ].reshape((1,)+(self.label_shape, self.label_shape)+(1,))

                            if self.custom_loss:
                                weights_ = weights[
                                    i*self.overlap: self.label_shape + i*self.overlap,
                                    j*self.overlap: self.label_shape + j*self.overlap
                                ].reshape((1,)+(self.label_shape, self.label_shape)+(1,))
                                img_ = [img_, weights_]

                            yield (img_, label_)
                        else:
                            yield img_
