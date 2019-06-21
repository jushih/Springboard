#!/Users/julieshih/anaconda/bin/python
from src.processing.utils import * 
import src.processing.config as cfg
from keras.models import model_from_json
from keras.models import load_model
import pickle
import pdb
import json

img_dir, metadata_dir, model_dir, search_img_dir = set_paths(cfg.PATH)

print('Loading trained model...')

vae = model_from_json(open(model_dir+'vggVAE.json').read())
vae.load_weights(model_dir+'vggVAE_weights.h5')

print(vae.summary())

with open(model_dir+'vggVAE_encoded_closet', 'rb') as ef:   
     encodings = pickle.load(ef)

with open(model_dir+'vggVAE_original_closet', 'rb') as of:
     originals = pickle.load(of)

retrieved = retrieve(vae.layers[1], originals, encodings, search_img_dir, target_size=cfg.IMAGE_SIZE, n=5)
print(retrieved)


