#!/Users/julieshih/anaconda/bin/python
from src.processing.utils import * 
import src.processing.config as cfg
from keras.models import model_from_json
from keras.models import load_model
from keras import Model, Input
import pickle
import os
import re
import json
import tensorflow as tf
from flask import Flask, request, Response, render_template, send_from_directory
import jsonpickle
from keras import backend as K

K.clear_session()

img_dir, metadata_dir, model_dir, search_img_dir = set_paths(cfg.PATH)

print('Loading trained model...')

#autoencoder = model_from_json(open(model_dir+'cnn_3L.json').read())
#autoencoder.load_weights(model_dir+'cnn_3L_weights.h5')
autoencoder = load_model(model_dir+'cnn_3L.h5')

# build encoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_output_at(0))

graph = tf.get_default_graph()

with open(model_dir+'vgg_encoded_closet', 'rb') as ef:   
     encodings = pickle.load(ef)


search_img_dir = '/Users/julieshih/workspace/Springboard/src/uploads/'

app = Flask(__name__, static_folder='/Users/julieshih/workspace/Springboard/data/img',root_path='src/')

# define apps home page
@app.route('/')
def index():
    return render_template('index.html')

# build API
@app.route('/uploads',methods=['POST'])
def upload():
    global graph
    with graph.as_default():

        # delete previous uploads so the generator doesn't read them
        for file in os.listdir('src/uploads/img/'):
            os.remove('src/uploads/img/'+file)

        # upload image
        upload_dir = 'src/uploads/img/' 

        if not os.path.isdir(upload_dir):
            os.mkdir(upload_dir)

        for img in request.files.getlist('file'):
            img_name = img.filename
            destination = '/'.join([upload_dir, img_name])
            img.save(destination)

        print(destination, upload_dir)

        print(encoder)
        # generate prediction and return similar images
        retrieved = retrieve(encoder, encodings, search_img_dir, target_size=cfg.IMAGE_SIZE, n=5)
       
        print(retrieved)
        
        filenames = []
        for filename in retrieved:
            filenames.append(re.split('/data/img/',filename)[1])
            
        print(filenames)

        response = {}
        response['retrieved'] = filenames
        response_pickled = jsonpickle.encode(response)

        return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/src/uploads/img/<filename>')
def send_image(filename):
    return send_from_directory('/Users/julieshih/workspace/Springboard/src/uploads/img/', filename)


if __name__ == "__main__":
    app.run(port=3000, debug=True)

