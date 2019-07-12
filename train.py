from __future__ import absolute_import, division, print_function
import os

import tensorflow as tf

from model_definition import create_model
from config import EPOCHS, BATCH_SIZE, keras_model_dir, serving_model_dir
from prepare_data import get_datasets

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

train_generator, valid_generator, \
train_num, valid_num, \
class_weights = get_datasets()

# start training
model = create_model()

model.fit_generator(train_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=train_num // BATCH_SIZE,
                    validation_data=valid_generator,
                    validation_steps=valid_num // BATCH_SIZE,
                    class_weight=class_weights
                    )

# save the whole keras model
model.save(os.path.join('keras_model', keras_model_dir))
# save saved_model for tensorflow serving
tf.keras.experimental.export_saved_model(model, os.path.join('saved_model', serving_model_dir))
