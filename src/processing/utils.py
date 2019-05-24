#!/Users/julieshih/.local/share/virtualenvs/fashion/bin/python
import os
import sys
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

# crawls image directory and saves a file of image paths, takes image folder and file location as inputs
def get_files(path, output_path):

  print('Crawling', path , '...' )

  f = []
  for root, _, filenames in os.walk(path):
    for filename in filenames:
      f.append(os.path.join(root, filename))

  paths_df = pd.DataFrame({'files':f})
  paths_df.to_csv(output_path+'paths.csv',index=False)
  print('Saved results to ',output_path+'paths.csv')


# reads file of image paths and returns a sample dataframe, takes file path and sample size as inputs
def load_data(path, sample_size=2000):

    paths_df = pd.read_csv(path, skiprows=3, header=None) # skip the first 3 non-path files
    files = paths_df[0].tolist()

    print('Number of images:', len(files))

    #filter images
    dress_files = []

    for path in files:
        #add filters here at a later stage e.g., if 'Dress' in path
        dress_files.append(path)

    df = pd.DataFrame(dress_files, columns=['filename'])

    df['folder'] = df.filename.astype(str).str.split('data/img').str[1].str.split('/').str[1]
    df['label'] = df.folder.astype(str).str.split('_')

    print(df.head())
    # Randomly subsample the data to get n rows
    df_sample = df.sample(n=sample_size, random_state=1)
    df_sample = df.sample.reset_index()

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
def split_data(df, directory, target_size, train_batch_size=32, test_batch_size=1, seed=42):

    train, validate, test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])

    datagen=ImageDataGenerator(rescale=1./255.)
    test_datagen=ImageDataGenerator(rescale=1./255.)

    train_generator=datagen.flow_from_dataframe(
      dataframe=train,
      directory=directory,
      x_col="filename",
      y_col="label",
      batch_size=train_batch_size,
      seed=seed,
      shuffle=True,
    #  classes = class_list,
      class_mode='input', #give input images as the labels as well since you are creating an autoencoder
      target_size=target_size)

    valid_generator=test_datagen.flow_from_dataframe(
      dataframe=validate,
      directory=directory,
      x_col="filename",
      y_col="label",
      batch_size=train_batch_size,
      seed=seed,
      shuffle=True,
    #  classes = class_list,
      class_mode="input",
      target_size=target_size)


    test_generator=test_datagen.flow_from_dataframe(
      dataframe=test,
      directory=directory,
      x_col="filename",
      batch_size=test_batch_size,
      seed=seed,
      shuffle=False,
      class_mode=None,
      target_size=target_size)

    return train_generator, valid_generator, test_generator

#df, class_list, dense_output = load_data('/Users/julieshih/workspace/Springboard/paths.csv')
#train_generator, valid_generator, test_generator = split_data(df, directory='/Users/julieshih/workspace/Springboard/', target_size=(32,32))

