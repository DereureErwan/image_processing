"""
Implementation of U-Net convolutional neural network
The architecture can be found here :
https://arxiv.org/pdf/1505.04597.pdf?fbclid=IwAR0aX0VGI4vMIfr7RJlV_peUAEqbGiUFKNvLH2ZotS2HOpo8898XHz65vBQ
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend_config
from tensorflow.keras.layers import (Dropout, Conv2D, concatenate, Input,
                                     MaxPooling2D, UpSampling2D, Cropping2D,
                                     BatchNormalization)


epsilon = backend_config.epsilon
set_epsilon = backend_config.set_epsilon


class UNet:

    def __init__(
        self,
        input_shape=(572, 572, 1),
        dropout=0.3,
        batch_normalisation=False,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        custom_loss=True,
    ):
        self.input_shape = input_shape
        self.dropout = dropout
        self.batch_normalisation = batch_normalisation
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.custom_loss = custom_loss

    def __Conv2D(
        self,
        inputs,
        filters,
        kernel=(3, 3),
        activation='relu',
    ):
        """
        wrapper for Conv2D for readability
        """
        x = Conv2D(
            filters,
            kernel,
            activation=activation,
            kernel_initializer='he_normal',
        )(inputs)
        if self.batch_normalisation:
            x = BatchNormalization()(x)
        return x

    def __UpConv2D(self, inputs, filters, kernel=(2, 2)):
        x = Conv2D(
            filters,
            kernel,
            padding='same',
            kernel_initializer='he_normal',
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
                (right_crop, left_crop),
                ))(conv), upsam
            ], axis=3
        )

    @staticmethod
    def __weighted_b_crossentropy(weights):
        def weighted_binary_crossentropy(y_true, y_pred):
            bce = y_true * tf.math.log(y_pred + epsilon())
            bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon())
            bce *= weights
            return tf.math.reduce_mean(- bce, axis=-1)
        return weighted_binary_crossentropy

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
        images = Input(shape=self.input_shape)
        loss_weights = Input(shape=self.input_shape)

        """""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""" Contracting Path """""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""

        conv1 = self.__Conv2D(images, 64)
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
            activation='softmax',
        )

        """""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""" Optimization """""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""

        model = Model(inputs=[images, loss_weights], outputs=conv10)
        optimizer = Adam(
            learning_rate=self.lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        # See here : https://github.com/tensorflow/tensorflow/issues/32142
        # XXX: custom loss not tested yet
        loss = UNet.__weighted_b_crossentropy(loss_weights) if self.custom_loss\
            else 'binary_cross_entropy'
        model.compile(
            optimizer,
            loss=loss,
            metrics=['accuracy'],
            experimental_run_tf_function=False,
        )

        return model
