"""
Hourglass-like model for denoising
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dropout, Conv2D, concatenate, Input,
                                     MaxPooling2D, UpSampling2D)


inputs = Input(shape=(100, 100, 1))

conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
drop3 = Dropout(0.5)(conv3)

up4 = Conv2D(128, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop3))
conc4 = concatenate([conv2, up4], axis=3)

conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc4)
conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

up6 = Conv2D(64, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
conc6 = concatenate([conv1, up6], axis=3)

conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc6)
conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

conv8 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7)

model = Model(input=inputs, output=conv8)
