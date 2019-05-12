#!/Users/julieshih/anaconda/bin/python
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import regularizers, optimizers
from keras import Model, Input

def cnn_3L():

    input_img = Input(shape=(32,32,3))
    x = Conv2D(64,(3,3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(32,(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(16,(3,3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    #Use three filters in the last layer since your images are RGB:
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoder')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

