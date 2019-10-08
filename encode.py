#!/Users/julieshih/anaconda/bin/python
from src.processing.utils import *
import src.processing.config as cfg
from keras.models import model_from_json
from keras.models import load_model
import pickle
import pdb
import json
from keras import Model, Input
"""Create embeddings with the trained model"""
img_dir, metadata_dir, model_dir, search_img_dir = set_paths(cfg.PATH)

print('Loading trained model...')

autoencoder = model_from_json(open(model_dir+'cnn_3L.json').read())
autoencoder.load_weights(model_dir+'cnn_3L_weights.h5')
autoencoder = load_model(model_dir+'cnn_3L.h5')

# build encoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_output_at(0))
print(encoder.summary())


df, class_list, dense_output = load_data(metadata_dir, sample_size=cfg.SAMPLE_SIZE)
inventory_generator = inventory_gen(df=df, img_dir=img_dir,target_size=cfg.IMAGE_SIZE)

# using the trained model, encode all clothing images for later use in knn retrieval
encodings = clothes_db(model_encoder=encoder,inventory_generator=inventory_generator, df=df)


print('pickling...')

with open(model_dir+'vgg_encoded_closet', 'wb') as f:
    pickle.dump(encodings, f, protocol=pickle.HIGHEST_PROTOCOL)


print('pickled')
