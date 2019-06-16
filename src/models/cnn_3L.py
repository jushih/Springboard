#!/Users/julieshih/anaconda/bin/python
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import regularizers, optimizers
from keras import Model, Input
import pdb


class AutoEncoder:
    def __init__(self):
        print('Initializing Autoencoder...')

    def _encoder(self):
        input_img = Input(shape=(32,32,3))
        x = Conv2D(64,(3,3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(32,(3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Conv2D(16,(3,3), activation='relu', padding='same')(x)

        encoded = MaxPooling2D((2,2), padding='same', name='encoder_layer')(x)
        
        model = Model(inputs=input_img, outputs=encoded, name='encoder')
        self.encoder = model
        return model

    def _decoder(self):
               
        encoded = Input(shape=(4,4,16))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        #Use three filters in the last layer since your images are RGB:
        decoded = Conv2D(3, (3, 3), activation='relu', padding='same', name='decoder_layer')(x)
    
        model = Model(inputs=encoded, outputs=decoded, name='decoder')
        self.decoder = model
        return model


    def encoder_decoder(self, optimizer='adam', loss='binary_crossentropy'):
        ec = self._encoder()
        dc = self._decoder()
        
        inputs = Input(shape=(32,32,3))
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
        self.model.save(model_dir+'cnn_3L.h5')
    
        # save model json
        model_json = self.model.to_json()
        with open(model_dir+'cnn_3L.json', 'w') as json_file:
            json_file.write(model_json)
       

#if __name__ == '__main__':
#    ae = AutoEncoder()
#    ae.encoder_decoder()
#    pdb.set_trace()
#    ae.fit(batch_size=50, epochs=300)
#    print(ae.model.summary())


