# training parameters
DATASET = 'flower_photos'
NUM_CLASSES = 5

EPOCHS = 100
BATCH_SIZE = 32
image_height = 224
image_width = 224
channels = 3
keras_model_dir = DATASET + "_mobilenetv2_model.h5"
serving_model_dir = DATASET + '_serving'
train_dir = "/home/ubuntu/dataset/{}/train".format(DATASET)
valid_dir = "/home/ubuntu/dataset/{}/val".format(DATASET)
