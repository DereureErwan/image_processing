"""
Implementing Data generator for U-Net feating
"""
import numpy as np
from scipy import ndimage
from skimage import io
import elasticdeform


class Data:
    """
    Class for data augmentation
    """
    TRAIN_DATA = 'data/train-volume.tif'
    TRAIN_LABELS = 'data/train-labels.tif'
    TEST_DATA = 'data/test-volume.tif'

    def __init__(
        self,
        shuffle=False,
        sigma=25,
        points=3,
        force_zero=True,
        rotate=None,
        tile=267,
        label_shape=68,
        overlap=37,
        extension=199,
        iteration=12
    ):
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

    def _data_augmentation(self, X, y):
        if (self.sigma is not None) and (self.points is not None):
            X, y = elasticdeform.deform_random_grid(
                [X, y],
                sigma=self.sigma, points=self.points
            )
        if self.rotate is not None:
            X, y = ndimage.rotate(X, self.rotate), ndimage.rotate(y, self.rotate)
        if self.force_zero:
            y = np.where(y <= 0.5, 0, 1)
        return X, y

    def _extend(self, X, extension=199):
        X_ = np.zeros((X.shape[0] + 2*extension, X.shape[1] + 2*extension))
        X_[extension: X.shape[0] + extension, extension: X.shape[1] + extension] = X
        # Complete X_ at left and right
        X_[extension:extension+X.shape[0], :extension] =\
            np.fliplr(X[:, :extension])
        X_[extension:extension+X.shape[0], -extension:] =\
            np.fliplr(X[:, -extension:])
        # Complete X_ at up and down
        X_[:extension, extension: extension+X.shape[1]:] =\
            np.flipud(X[:extension, :])
        X_[-extension:, extension: extension+X.shape[1]:] =\
            np.flipud(X[-extension:, :])
        # Complete corners
        # left-up
        X_[:extension, :extension] = np.flip(X[:extension, :extension])
        # left-down
        X_[-extension:, :extension] = np.flip(X[-extension:, :extension])
        # right-up
        X_[:extension, -extension:] = np.flip(X[:extension, -extension:])
        # right-down
        X_[-extension:, -extension:] = np.flip(X[-extension:, -extension:])
        return X_

    def generator(self, train=True):
        # TODO: Accept batch size of more than one
        idx = np.arange(0, self.n_samples)
        if self.shuffle:
            idx = np.random.shuffle(idx)
        for s in range(0, self.n_samples):
            img = self.train[s]
            if train:
                label = self.labels[s]
                img, label = self._data_augmentation(img, label)
            img = self._extend(img, self.extension)
            for i in range(self.iteration):
                for j in range(self.iteration):
                    img_ = img[
                        i*self.overlap: self.tile + i*self.overlap,
                        j*self.overlap: self.tile + j*self.overlap
                    ]
                    if train:
                        label_ = label[
                            i*self.overlap: self.label_shape + i*self.overlap,
                            j*self.overlap: self.label_shape + j*self.overlap
                        ]
                        yield (
                            img_.reshape((1,)+img_.shape+(1,)),
                            label_.reshape((1,)+label_.shape+(1,))
                        )
                    else:
                        yield img_.reshape((1,)+img_.shape+(1,))

    def _compute_weights(self, y):
        """
        compute weights for each label image
        as indicated in the U-Net paper
        """
        # XXX: not working
        raise NotImplementedError
        class_weights = np.mean(y)
        size = y.shape[0]
        y *= -1
        label = 1
        for i in range(size):
            for j in range(size):
                if y[i, j] == -1:
                    try:
                        y[i, j] = list(
                            set(y[i-1:i+2, j-1:j+2].flatten())
                            .difference({0, -1})
                            )[0]
                    except IndexError:
                        y[i, j] = label
                        label += 1
