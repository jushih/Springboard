#!/Users/julieshih/anaconda/bin/python
from src.processing.utils import * 
import src.processing.config as cfg
from src.models.cnn_3L import *
from src.models.vgg import *

img_dir, metadata_dir, model_dir, search_img_dir = set_paths(cfg.PATH)

get_files(img_dir=img_dir, metadata_dir=metadata_dir)

df, class_list, dense_output = load_data(metadata_dir, sample_size=cfg.SAMPLE_SIZE)

train_generator, valid_generator, test_generator = split_data(df, 
                                                              img_dir=img_dir, 
                                                              target_size=cfg.IMAGE_SIZE,
                                                              train_batch_size=cfg.TRAIN_BATCH_SIZE,
                                                              test_batch_size=cfg.TEST_BATCH_SIZE,
                                                              seed=cfg.SEED)

inventory_generator = inventory_gen(df, img_dir=img_dir,target_size=cfg.IMAGE_SIZE)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

vae = AutoEncoder()
vae.encoder_decoder(image_dim=cfg.IMAGE_DIM)

vae.fit( train_generator, 
         valid_generator, 
         STEP_SIZE_TRAIN, 
         STEP_SIZE_VALID,
         epochs = cfg.EPOCHS)

print(vae.model.summary())
vae.save(model_dir)


originals, encodings = clothes_db(model=vae,inventory_generator=inventory_generator)
print(encodings)

retrieved = retrieve(vae, originals, encodings, search_img_dir, target_size=cfg.IMAGE_SIZE, n=5)
print(retrieved)
