from collections import Counter

import tensorflow as tf

import config

def get_datasets():
    # Preprocess the dataset
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0, 
        shear_range=0.2,
        zoom_range=0.2,
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(config.train_dir,
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        # seed=1, change seed to none for random transformation each epoch
                                                        shuffle=True,
                                                        class_mode="categorical")
    class_weights = {}
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}

    print("class weights for alphapbet ordering class:", class_weights)

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 /255.0,
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    valid_generator = valid_datagen.flow_from_directory(config.valid_dir,
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )

    train_num = train_generator.samples
    valid_num = valid_generator.samples


    return train_generator, \
           valid_generator, \
           train_num, valid_num, \
           class_weights
