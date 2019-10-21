# Dress Retrieval App

The goal of this project is to build an image search app that can retrieve similar-looking dresses based on an existing image of a dress. To improve sales, imagine that a retailer wants their website to show users dresses similar to the one they've currently selected. Or imagine that a user has found or taken a picture of a dress they like and wants to buy something similar. With this Dress Retrieval App, users can upload a photo of their favorite dress and the app will return similar dresses available in stores. 

The algorithm powering the app was constructed by:

* Training a 3-layer CNN Autoencoder with Keras
* Using the encoder part of the Autoencoder model, vector representations (encodings) of each dress image was generated
* With the image encodings, k-means clustering was applied to sort images into one of 10 clusters
* When a photo is uploaded, the k-means model predicts the cluster the photo belongs to, and a K-Nearest Neighbor algorithm is then applied over all embeddings within the cluster to identify the most similar images

The training data was obtained from The Chinese University of Hong Kong’s DeepFashion Attribute Prediction database. It consists of 289,222 images from 50 clothing categories. In this project, I will use the 71,314 dress images for training the model. These dresses will also serve as the "in-store" database that the web app will search and retrieve from.

# File Structure

```
README.md
data
├── img
│   ├── folders
models
src
├── exploration
│   ├── exploration_attributes.ipynb
│   └── exploration_landmark.ipynb
├── processing
│   ├── config.py
│   └── utils.py
├── models
│   ├── cnn_3L.py
│   ├── vgg.py
├── scraper
│   ├── full_scraper.py
│   ├── scraper.ipynb
│   └── scraper.py
└── templates
    ├── index.html
    └── result.html
app.py
train.py
encode.py
Dockerfile
requirements.txt

```

**train.py** - trains the autoencoder model
**encode.py** - generates the image embeddings using the encoder part of the autoencoder
**app.py** - launches the web app where users can upload an image
**api.py** - launches the API
**data** - contains a sample of the training images
**models** - contains the trained models and image embeddings
**src/exploration** - initial exploration of the image dataset
**src/processing** - helper code to prep the dataset, run the model, and configure model parameters
**src/models** - specifies the model architecture
**src/scraper** - code to scrape more image data
**src/templates** - contains the html pages served by the web app

# API

```
# launches the API
python3 api.py

# send an image to an API and return the most similar dress
 curl -X POST -F "file=@/path/to/dress.jpg" http://127.0.0.1:2000/upload -o retrieved_dress.jpg
```

# References

Large-scale Fashion (DeepFashion) Database - http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

Liu, Ziwei, et al. "Deepfashion: Powering robust clothes recognition and retrieval with rich annotations." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

Liu, Ziwei, et al. "Fashion landmark detection in the wild." European Conference on Computer Vision. Springer, Cham, 2016.

