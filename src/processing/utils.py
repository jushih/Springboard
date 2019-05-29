#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import os
import sys
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import pdb

# returns paths depending on working location
def set_paths(loc):
     
    if loc == "local":
        img_dir = '/Users/julieshih/workspace/Springboard/data/img/'
        metadata_dir = '/Users/julieshih/workspace/Springboard/data/Anno/'
        return img_dir, metadata_dir

    if loc == "docker":
        img_dir = '/fashion/data/img/'
        metadata_dir = '/fashion/data/Anno/'
        return img_dir, metadata_dir


    if loc == "colab":
        img_dir = '/content/gdrive/My Drive/data/img/'
        metadata_dir = '/content/gdrive/My Drive/data/Anno/'
        return img_dir, metadata_dir

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

    #filter images
    dress_files = []

    for path in files:
        #add filters here at a later stage e.g., if 'Dress' in path
        dress_files.append(path)
    
    df = pd.DataFrame(dress_files, columns=['filename'])

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




#df, class_list, dense_output = load_data('/Users/julieshih/workspace/Springboard/paths.csv')
#train_generator, valid_generator, test_generator = split_data(df, directory='/Users/julieshih/workspace/Springboard/', target_size=(32,32))

