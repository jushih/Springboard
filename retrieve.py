#!/Users/julieshih/anaconda/bin/python
from src.processing.utils import * 
import src.processing.config as cfg
from keras.models import model_from_json
from keras.models import load_model
from keras import Model, Input
import pickle
import pdb
import json
from flask import Flask, request, render_template, send_from_directory

img_dir, metadata_dir, model_dir, search_img_dir = set_paths(cfg.PATH)

print('Loading trained model...')

autoencoder = model_from_json(open(model_dir+'cnn_3L.json').read())
autoencoder.load_weights(model_dir+'cnn_3L_weights.h5')
autoencoder = load_model(model_dir+'cnn_3L.h5')

# build encoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_output_at(0))
print(encoder.summary())


with open(model_dir+'vgg_encoded_closet', 'rb') as ef:   
     encodings = pickle.load(ef)

print(encodings)

print(search_img_dir)
search_img_dir = '/Users/julieshih/workspace/Springboard/src/uploads/'

retrieved = retrieve(encoder, encodings, search_img_dir, target_size=cfg.IMAGE_SIZE, n=5)
print(retrieved)
