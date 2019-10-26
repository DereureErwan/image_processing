"""
Implementation of U-Net convolutional neural network
The architecture can be found here :
https://arxiv.org/pdf/1505.04597.pdf?fbclid=IwAR0aX0VGI4vMIfr7RJlV_peUAEqbGiUFKNvLH2ZotS2HOpo8898XHz65vBQ
"""
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import (Dropout, Conv2D, concatenate, Input,
                                     MaxPooling2D, UpSampling2D, Cropping2D,
                                     BatchNormalization)


class UNet:

    def __init__(
        self,
        input_shape=(572, 572, 1),
        lr=0.001,
        momentum=0.99,
        dropout=0.3,
        batch_normalisation=False
    ):
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.dropout = dropout
        self.batch_normalisation = batch_normalisation

    def __Conv2D(
        self,
        inputs,
        filters,
        kernel=(3, 3),
        activation='relu'
    ):
        """
        wrapper for Conv2D for readability
        """
        x = Conv2D(
            filters,
            kernel,
            activation=activation,
            kernel_initializer='he_normal'
            )(inputs)
        if self.batch_normalisation:
            x = BatchNormalization()(x)
        return x

    def __UpConv2D(self, inputs, filters, kernel=(2, 2)):
        x = Conv2D(
            filters,
            kernel,
            padding='same',
            kernel_initializer='he_normal'
            )(UpSampling2D(size=(2, 2))(inputs))

        if self.batch_normalisation:
            x = BatchNormalization()(x)
        return x

    def __copycrop_and_concatenate(self, conv, upsam):
        crop = (conv.shape[1] - upsam.shape[1])
        right_crop, left_crop = crop//2 + crop % 2, crop//2
        return concatenate([
            Cropping2D(cropping=(
                (right_crop, left_crop),
                (right_crop, left_crop)
                ))(conv), upsam
            ], axis=3
        )

    def __call__(self):
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
        inputs = Input(shape=self.input_shape)

        """""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""" Contracting Path """""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""

        conv1 = self.__Conv2D(inputs, 64)
        conv1 = self.__Conv2D(conv1, 64)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        drop1 = Dropout(self.dropout)(pool1)

        conv2 = self.__Conv2D(drop1, 128)
        conv2 = self.__Conv2D(conv2, 128)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        drop2 = Dropout(self.dropout)(pool2)

        conv3 = self.__Conv2D(drop2, 256)
        conv3 = self.__Conv2D(conv3, 256)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        drop3 = Dropout(self.dropout)(pool3)

        conv4 = self.__Conv2D(drop3, 512)
        conv4 = self.__Conv2D(conv4, 512)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        drop4 = Dropout(self.dropout)(pool4)

        conv5 = self.__Conv2D(drop4, 1024)
        conv5 = self.__Conv2D(conv5, 1024)
        upsam5 = self.__UpConv2D(conv5, 512)

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""" Expansive Path """""""""""""""""""""""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""

        conc6 = self.__copycrop_and_concatenate(conv4, upsam5)
        conv6 = self.__Conv2D(conc6, 512)
        conv6 = self.__Conv2D(conv6, 512)
        upsam6 = self.__UpConv2D(conv6, 512)

        conc7 = self.__copycrop_and_concatenate(conv3, upsam6)
        conv7 = self.__Conv2D(conc7, 256)
        conv7 = self.__Conv2D(conv7, 256)
        upsam7 = self.__UpConv2D(conv7, 512)

        conc8 = self.__copycrop_and_concatenate(conv2, upsam7)
        conv8 = self.__Conv2D(conc8, 128)
        conv8 = self.__Conv2D(conv8, 128)
        upsam8 = self.__UpConv2D(conv8, 512)

        conc9 = self.__copycrop_and_concatenate(conv1, upsam8)
        conv9 = self.__Conv2D(conc9, 64)
        conv9 = self.__Conv2D(conv9, 64)

        conv10 = self.__Conv2D(
            conv9,
            1, kernel=(1, 1),
            activation='softmax'
            )

        """""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""" Optimization """""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""

        model = Model(inputs=inputs, outputs=conv10)
        optimizer = SGD(learning_rate=self.lr, momentum=self.momentum, nesterov=True)
        model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model
