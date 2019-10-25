"""
Implementation of U-Net convolutional neural network
The architecture can be found here :
https://arxiv.org/pdf/1505.04597.pdf?fbclid=IwAR0aX0VGI4vMIfr7RJlV_peUAEqbGiUFKNvLH2ZotS2HOpo8898XHz65vBQ
"""
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Dropout, Conv2D, concatenate, Input,
                                     MaxPooling2D, UpSampling2D, Cropping2D)


def __stddev(x):
    """
    Initialize weight in convolution layers as
    indicate in the U-Net paper

    """
    # TODO: compute x with previous layer
    return RandomNormal(stddev=(2/x)**0.5)


def __Conv2D(filters, kernel=(3, 3), activation='relu', std=1):
    """
    wrapper for Conv2D for readability
    """
    return Conv2D(
        filters,
        kernel,
        activation=activation,
        kernel_initializer=__stddev(std)
        )


def create_model(lr=0.001, momentum=0.99, dropout=0.3):
    """
    Defines our U-Net

    Args:
        lr: float, default=0.001
            learning rate for SGD
        momentum: float: default=0.99
            momentum for SGD
        dropout: float, default=0.3
            probability of dropout for contracting path
    """
    inputs = Input(shape=(512, 512, 1))

    """""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""" Contracting Path """""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""

    conv1 = __Conv2D(64, std=3*3*512)(inputs)
    conv1 = __Conv2D(64, std=3*3*64)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = __Conv2D(128, std=3*3*64)(Dropout(dropout)(pool1))
    conv2 = __Conv2D(128, std=3*3*128)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = __Conv2D(256, std=3*3*128)(Dropout(dropout)(pool2))
    conv3 = __Conv2D(256, std=3*3*256)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = __Conv2D(512, std=3*3*256)(Dropout(dropout)(pool3))
    conv4 = __Conv2D(512, std=3*3*512)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = __Conv2D(1024, std=3*3*512)(Dropout(dropout)(pool4))
    conv5 = __Conv2D(1024, std=3*3*1024)(conv5)
    upsam5 = __Conv2D(512, kernel=(2, 2), std=3*3*1024)(UpSampling2D(size=(2, 2))(conv5))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""" Expansive Path """""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""

    conc6 = concatenate([
        Cropping2D(cropping=((4, 5), (4, 5)))(conv4), upsam5
        ], axis=3)
    conv6 = __Conv2D(512, std=2*2*512)(conc6)
    conv6 = __Conv2D(512, std=3*3*512)(conv6)
    up6 = __Conv2D(256, kernel=(2, 2), std=3*3*512)(UpSampling2D(size=(2, 2))(conv6))

    conc7 = concatenate([
        Cropping2D(cropping=((18, 18), (18, 18)))(conv3), up6
        ], axis=3)
    conv7 = __Conv2D(256, std=2*2*256)(conc7)
    conv7 = __Conv2D(256, std=3*3*256)(conv7)
    up7 = __Conv2D(128, kernel=(2, 2), std=3*3*256)(UpSampling2D(size=(2, 2))(conv7))

    conc8 = concatenate([
        Cropping2D(cropping=((44, 45), (44, 45)))(conv2), up7
        ], axis=3)
    conv8 = __Conv2D(128, std=2*2*128)(conc8)
    conv8 = __Conv2D(128, std=3*3*128)(conv8)
    up8 = __Conv2D(64, kernel=(2, 2), std=3*3*128)(UpSampling2D(size=(2, 2))(conv8))

    conc9 = concatenate([
        Cropping2D(cropping=((97, 98), (97, 98)))(conv1), up8
        ], axis=3)
    conv9 = __Conv2D(64, std=2*2*64)(conc9)
    conv9 = __Conv2D(64, std=3*3*64)(conv9)

    conv10 = __Conv2D(
        1, kernel=(1, 1),
        activation='softmax',
        std=3*3*64
        )(conv9)

    """""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""" Optimization """""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""

    model = Model(inputs=inputs, outputs=conv10)
    optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=True)
    model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
