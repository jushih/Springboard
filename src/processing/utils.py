#!/Users/julieshih/.local/share/virtualenvs/fashion/bin/python
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
path = sys.argv[1]
#path = 'gdrive/My Drive/img_attributes/Anno/paths.csv'

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


def split_data(df, directory, target_size, train_batch_size=32, test_batch_size=1, seed=42):

    datagen=ImageDataGenerator(rescale=1./255.)
    test_datagen=ImageDataGenerator(rescale=1./255.)

    train_generator=datagen.flow_from_dataframe(
      dataframe=df.iloc[:1800],
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
      dataframe=df.iloc[1800:1900],
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
      dataframe=df.iloc[1900:],
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

