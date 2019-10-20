#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import os
import sys
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from imageio import imread
import matplotlib.image as mpimg
import pdb

# returns paths depending on working location
def set_paths(loc):
     
    if loc == "local":
        img_dir = '/Users/julieshih/workspace/Springboard/data/img/'
        metadata_dir = '/Users/julieshih/workspace/Springboard/data/Anno/'
        model_dir = '/Users/julieshih/workspace/Springboard/models/'
        search_img_dir = '/Users/julieshih/workspace/Springboard/src/uploads/'
        return img_dir, metadata_dir, model_dir, search_img_dir

    if loc == "docker":
        img_dir = '/fashion/data/img/'
        metadata_dir = '/fashion/data/Anno/'
        model_dir = '/fashion/models/'
        search_img_dir = 'fashion/data/search_img/'
        return img_dir, metadata_dir, model_dir, search_img_dir

    if loc == "colab":
        img_dir = '/content/gdrive/My Drive/data/img/'
        metadata_dir = '/content/gdrive/My Drive/data/Anno/'
        model_dir = '/content/gdrive/My Drive/data/Anno/'
        search_img_dir = '/content/gdrive/My Drive/data/search_img/'
        return img_dir, metadata_dir, model_dir, search_img_dir

    # add aws paths eventually


# crawls image directory and saves a file of image paths to metadata folder
def get_files(img_dir, metadata_dir):

  print('Crawling', img_dir , '...' )

  f = []
  for root, _, filenames in os.walk(img_dir):
    for filename in filenames:
      f.append(os.path.join(root, filename))

  paths_df = pd.DataFrame({'files':f})
  paths_df.to_csv(metadata_dir+'paths.csv',index=False)
  print('Saved results to ',metadata_dir+'paths.csv')


# reads file of image paths and returns a sample dataframe, takes metadata folder and sample size as inputs
def load_data(metadata_dir, sample_size=2000):

    paths_df = pd.read_csv(metadata_dir+'paths.csv', skiprows=3, header=None) # skip the first 3 non-path files
    files = paths_df[0].tolist()

    print('Number of images:', len(files))

    print('Filtering to dresses...')
    #filter images
    dress_files = []

    for path in files:
        if 'Dress' in path:
            dress_files.append(path)
    
    df = pd.DataFrame(dress_files, columns=['filename'])

    print('Number of dress images:', df.shape[0])

    df['folder'] = df.filename.astype(str).str.split('/data/img').str[1].str.split('/').str[1]
    df['label'] = df.folder.astype(str).str.split('_')

    print(df.head())
    # Randomly subsample the data to get n rows
    df_sample = df.sample(n=sample_size, random_state=1)
    df_sample = df_sample.reset_index()

    print('Sampled',df_sample.shape[0],'images')

    # create a unique list of classes
    classes = []
    for index, row in df_sample.iterrows():
        for line in row['label']:
            classes.append(line)
    class_list = set(classes)

    # find total number of unique classes and save as dense_output
    dense_output = len(class_list)
    print('There are', len(class_list), 'classes.')

    return df_sample, class_list, dense_output

# takes a dataframe of image paths and splits it for training
def split_data(df, img_dir, target_size, train_batch_size=10, test_batch_size=1, seed=42):

    train, validate, test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])

    datagen=ImageDataGenerator(rescale=1./255.)
    test_datagen=ImageDataGenerator(rescale=1./255.)

    train_generator=datagen.flow_from_dataframe(
      dataframe=train,
      directory=img_dir,
      x_col="filename",
 #     y_col="label",
      batch_size=train_batch_size,
      seed=seed,
      shuffle=True,
    #  classes = class_list,
      class_mode='input', #give input images as the labels as well since you are creating an autoencoder
      target_size=target_size)

    valid_generator=test_datagen.flow_from_dataframe(
      dataframe=validate,
      directory=img_dir,
      x_col="filename",
#      y_col="label",
      batch_size=train_batch_size,
      seed=seed,
      shuffle=True,
    #  classes = class_list,
      class_mode="input",
      target_size=target_size)


    test_generator=test_datagen.flow_from_dataframe(
      dataframe=test,
      directory=img_dir,
      x_col="filename",
      batch_size=test_batch_size,
      seed=seed,
      shuffle=False,
      class_mode=None,
      target_size=target_size)

    return train_generator, valid_generator, test_generator



def inventory_gen(df, img_dir, target_size=(32,32), seed=42, batch_size=1):
    
    datagen=ImageDataGenerator(rescale=1./255.)
    
    inventory_generator=datagen.flow_from_dataframe(
        dataframe=df,
        directory=img_dir,
        x_col="filename",
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        class_mode=None,
        target_size=target_size)

    return inventory_generator

# use the trained model to create encodings of each image
def clothes_db(model_encoder, inventory_generator, df):

    encodings = {}  # dictionary of encodings mapped to filenames
    n = 0

    for i in range(0,inventory_generator.n):


        # save pathnames as dict key
        img_path = df.filename.iloc[i]

        # save embeddings as dict value
        im = inventory_generator[i] 
        encoded = model_encoder.predict(im)
        squeezed = np.squeeze(encoded,axis=0)
  
        reshaped = np.hstack(np.hstack(squeezed)) # stack twice to reduce dimensions from (x, x, x) to (x,)
        encodings[img_path] = reshaped

        #n += 1
        #print(n)
    
    return encodings

# cluster the encoded images
def cluster(encodings):
    
    encoded_values = list(encodings.values())
    kmeans_clf = KMeans(n_clusters=10, random_state=0).fit(encoded_values)

    # get cluster labels
    labels = kmeans_clf.predict(list(encodings.values()))

    # create a dict of filename and cluster label
    clusters = dict(zip(list(encodings.keys()), labels))

    # create a new dict that has filename, label, and encoding
    # format {filename: [encoding], label]}
    db  = {}
    for key in (encodings.keys() | clusters.keys()):
        if key in encodings: db.setdefault(key, []).append(encodings[key])
        if key in clusters: db.setdefault(key, []).append(clusters[key])

    print(db)

    return kmeans_clf, db
    
# returns similar images by first using kmeans to find images belonging the cluster
# then using knn to retrieve most similar images
def retrieve(model_encoder, db, kmeans_clf, search_img_dir, target_size=(32,32), n=6):

    # load search img and rescale, this generator wil contain only the search img
    datagen=ImageDataGenerator(rescale=1./255.)
   
    search_img=datagen.flow_from_directory(
        directory=search_img_dir,
        batch_size=1,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=target_size)
    
    encoded_search_img = model_encoder.predict(search_img[0])
    squeezed = np.squeeze(encoded_search_img,axis=0)
    # stack twice to reduce dimensions from (x, x, x) to (x,) then reshape to (1,x)
    reshaped_search_img = np.hstack(np.hstack(squeezed)).reshape(1,-1)  

    # get the cluster the search image belongs to
    label = kmeans_clf.predict(encoded_search_img.reshape(encoded_search_img.shape[0],-1))
    label = label[0]

    # build dict of encodings based on images in the cluster
    encodings = {}
    for key, value in db.items():

        # value[0] are encodings, value[1] are clusters
        if value[1] == label:
            encodings[key] = value[0]


    print(len(encodings), ' images in this cluster.')

    # fit knn model to encodings
    nbrs = NearestNeighbors(n_neighbors=n).fit(list(encodings.values()))

    # retrieve nearest images
    distances, indices = nbrs.kneighbors(reshaped_search_img)
    print(indices)

    # later add distances cutoff here

    return_imgs = []
    for i in indices[0]:
        #print('Retrieved clothing items ' + str(list(encodings.keys())[i]) + ' as nearest match.')
        return_imgs.append(list(encodings.keys())[i])

       
    return return_imgs

