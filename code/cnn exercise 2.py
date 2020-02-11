'''
TU/e BME Project Imaging 2019
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

import os

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D, Conv1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc, roc_auc_score

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
    # dataset parameters
    train_path = os.path.join(base_dir, 'train+val', 'train')
    valid_path = os.path.join(base_dir, 'train+val', 'valid')

    RESCALING_FACTOR = 1. / 255

    # instantiate data generators
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=train_batch_size,
                                            class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=val_batch_size,
                                          class_mode='binary')

    return train_gen, val_gen


def get_model(kernel_size=(3, 3), pool_size=(4, 4), first_filters=32, second_filters=64):
    # build the model
    model = Sequential()

    model.add(
        Conv2D(first_filters, kernel_size, activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(64, (6, 6), activation='relu', padding='valid'))
    model.add(Conv2D(1, (1, 1), activation='relu', padding='valid'))


    # compile the model
    model.compile(SGD(lr=0.01, momentum=0.95), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# get the model
model = get_model()
for layer in model.layers:
    print(layer.output_shape)

# get the data generators
train_gen, val_gen = get_pcam_generators('C:/Users/max/stack/TUE/Sync_laptop/Imaging project/.data')

# save the model and weights
model_name = 'my_first_cnn_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json()  # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)

# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]

# train the model
train_steps = train_gen.n // train_gen.batch_size
val_steps = val_gen.n // val_gen.batch_size

history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                              validation_data=val_gen,
                              validation_steps=val_steps,
                              epochs=3,
                              callbacks=callbacks_list)

# ROC analysis
y_val_score = np.rint(model.predict_generator(val_gen, steps=val_steps))
y_val_true = val_gen.classes
area_under_curve = roc_auc_score(y_val_score, y_val_true)
print(area_under_curve)
