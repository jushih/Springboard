#!/Users/julieshih/anaconda/bin/python
from src.processing.utils import * 
from src.models.cnn_3L import *

img_dir, metadata_dir = set_paths("docker")

get_files(img_dir=img_dir, metadata_dir=metadata_dir)

df, class_list, dense_output = load_data(metadata_dir, sample_size=100)

train_generator, valid_generator, test_generator = split_data(df, 
                                                              img_dir=img_dir, 
                                                              target_size=(32,32))

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

ae = AutoEncoder()
ae.encoder_decoder()
ae.fit(train_generator, valid_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID)
print(ae.model.summary())