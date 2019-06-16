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
import pdb

class vggVAE:

    def __init__(self):
        print('Initializing VGG-16 Autoencoder...')
        self.latent_dim = 64 # representation of the image

    def _encoder(self):

        # for sampling of the VAE latent vector
        def sampling(args):
            """Reparameterization trick by sampling fr an isotropic unit Gaussian.

            # Arguments
                args (tensor): mean and log of variance of Q(z|X)

            # Returns
                z (tensor): sampled latent vector
            """

            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean=0 and std=1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        # import the VGG model
        vgg16 = VGG16(weights=None, include_top=True,  input_shape=(32,32,3))
        vgg16.layers.pop() # remove last layer

        # freeze vgg16 layers
        for layer in vgg16.layers:
            layer.trainable = False

        # transition dense layer  
        x = Dense(512)(vgg16.layers[-1].output)  

        # construct the z layer that consists of the mean vector and std dev. vector
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        encoder = Model(input=vgg16.input, output=[z_mean, z_log_var, z], name='encoder')
        self.encoder = encoder
        return encoder

    def _decoder(self):

        # input the sampled vector, then build vgg backwards to create decoder
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
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

    def encoder_decoder(self, optimizer='adam'):
   
        # build VAE
        ec = self._encoder()
        dc = self._decoder()
        
        inputs = ec.input

        # remember the output of the encoder model is: output=[z_mean, z_log_var, z]
        z_mean = ec(inputs)[0] # for the loss model 
        z_log_var = ec(inputs)[1] # for the loss model
        z_vec = ec(inputs)[2] # pass the decoder the z vector

        decoded = dc(z_vec)
        model = Model(inputs=inputs, outputs=decoded)
 

        # same as input shape, modularize this later
        original_dim = 32*32

        # Compute VAE loss
        def my_vae_loss(y_true, y_pred, z_log_var=z_log_var, z_mean=z_mean):
            
            # input vs. reconstructed, crossentropy (pixel represented by 0 and 1)
            reconstruction_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
            reconstruction_loss *= original_dim
            
            # reparametization trick, allow for backpropogation
            # KL divergence, ensure images are from normal distribution
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            
            return vae_loss

            
        self.model = model
        self.model.compile(loss=my_vae_loss, optimizer=optimizer)
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
        self.model.save(model_dir+'vgg_vae.h5')

        # save model json
        model_json = self.model.to_json()
        with open(model_dir+'vgg_vae.json', 'w') as json_file:
            json_file.write(model_json)

#if __name__ == '__main__':
#    vae = vggVAE()
#    vae.encoder_decoder()
#    print(vae.model.summary())



