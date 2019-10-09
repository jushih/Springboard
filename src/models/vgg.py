#!/Users/julieshih/anaconda/bin/python
import keras
import tensorflow as tf
from keras.models import Sequential, Model
"""Import from keras_preprocessing not from keras.preprocessing, because Keras may or maynot contain the features discussed here depending upon when you read this article, until the keras_preprocessed library is updated in Keras use the github version."""
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras import regularizers, optimizers
from keras import Model, Input
import pandas as pd
import numpy as np
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
import pdb

class vgg:

    def __init__(self):
        print('Initializing VGG-16 Autoencoder...')

    def _encoder(self):

        # import the VGG model
        vgg16 = VGG16(weights=None, include_top=True,  input_shape=(32,32,3))
        vgg16.layers.pop() # remove last layer

        # freeze vgg16 layers
        for layer in vgg16.layers:
            layer.trainable = False

        # construct encoder
        encoder = Model(input=vgg16.input, output=vgg16.layers[-4].output, name='encoder')
        self.encoder = encoder
        return encoder

    def _decoder(self):

        # build vgg backwards to create decoder
        latent_inputs = Input(shape=(1,1,512))
        x = Dense(512, activation='sigmoid')(latent_inputs)
        x = Reshape((1,1,512))(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(512, (2, 2), padding='same')(x)
        x = Conv2D(512, (2, 2), padding='same')(x)
        x = Conv2D(512, (2, 2), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(512, (2, 2), padding='same')(x)
        x = Conv2D(512, (2, 2), padding='same')(x)
        x = Conv2D(256, (2, 2), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(256, (2, 2), padding='same')(x)
        x = Conv2D(256, (2, 2), padding='same')(x)
        x = Conv2D(128, (2, 2) ,padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = Conv2D(128, (2, 2), padding='same')(x)
        x = Conv2D(64, (2, 2) ,padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (2, 2), padding='same')(x)
        x = Conv2D(64, (2, 2), padding='same')(x)
        outputs = Conv2D(3, (2, 2) ,padding='same')(x)

        decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder = decoder
        return decoder

    def encoder_decoder(self, optimizer='adam', loss='binary_crossentropy'):
   
        # build AE
        ec = self._encoder()
        dc = self._decoder()
        
        #inputs = Input(shape=(32,32,3))
        inputs = ec.input
        encoded = ec(inputs)
        decoded = dc(encoded)

        model = Model(inputs=inputs, outputs=decoded)
            
        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss)
        return model


    def fit(self, train_generator, valid_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID, epochs=32):
#       tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,
                    verbose=1,
                    shuffle=True)


    def save(self, model_dir):
        # save h5 file
        self.model.save(model_dir+'vgg.h5')

        # save model weights
        self.model.save_weights(model_dir+'vgg_weights.h5')
        
        # save model json
        model_json = self.model.to_json()
        with open(model_dir+'vgg.json', 'w') as json_file:
            json_file.write(model_json)

         # save model architecture as image
        plot_model(self.encoder, to_file=model_dir+'vgg_encoder.png',show_shapes=True, show_layer_names=True)
        plot_model(self.decoder, to_file=model_dir+'vgg_decoder.png',show_shapes=True, show_layer_names=True)

#if __name__ == '__main__':
#    vgg = vgg()
#    vgg.encoder_decoder()
#    print(vgg.model.summary())


