#!/Users/julieshih/anaconda/bin/python
from src.processing.utils import * 
import src.processing.config as cfg
#from keras.models import model_from_json
#from keras.models import load_model
import pickle
import pdb
import json
from flask import Flask, request, render_template, send_from_directory

#img_dir, metadata_dir, model_dir, search_img_dir = set_paths(cfg.PATH)

#print('Loading trained model...')

#vae = model_from_json(open(model_dir+'vggVAE.json').read())
#vae.load_weights(model_dir+'vggVAE_weights.h5')

#print(vae.summary())

#with open(model_dir+'vggVAE_encoded_closet', 'rb') as ef:   
#     encodings = pickle.load(ef)

#with open(model_dir+'vggVAE_original_closet', 'rb') as of:
#     originals = pickle.load(of)

#retrieved = retrieve(vae.layers[1], originals, encodings, search_img_dir, target_size=cfg.IMAGE_SIZE, n=5)
#print(retrieved)


app = Flask(__name__, static_url_path='/static', root_path='src/')

# define apps home page
@app.route('/')
def index():
    return render_template('index.html')

# define upload function
@app.route('/upload',methods=['POST'])
def upload():

    upload_dir = 'src/uploads/' 

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    for img in request.files.getlist('file'):
        img_name = img.filename
        destination = '/'.join([upload_dir, img_name])
        img.save(destination)

    return_img = 'data/img_cut/Strappy_Floral-Embroidered_Dress/img_00000001.jpg'
    return render_template('result.html', image_name=img_name, result_paths=return_img)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory('src/uploads', filename)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
