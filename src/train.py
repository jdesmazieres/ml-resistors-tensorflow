"""
Model training and store
"""
import os
import glob
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

DATASET_ROOT_DIR = "../resources/images/"
MODEL_DIR = "model"
CHECKPOINT_PATH = MODEL_DIR + "/checkpoint-{epoch:04d}.ckpt"
MODEL_PATH = MODEL_DIR + "/resistors-model.h5"
BATCH_SIZE = 32
IMG_HEIGHT = 176
IMG_WIDTH = 64
SEED = 1337
AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASSES = ['10', '100', '100k', '10k', '150', '1M', '1k',
           '20', '200', '20k', '220', '220k', '270', '2k', '2k2',
           '300k', '330', '3k3',
           '470', '470k', '47k', '4k7',
           '510', '51k', '5k1',
           '680', '680k', '68k', '6k8']


# https://www.tensorflow.org/tutorials/load_data/images
def load(data_dir):
    print("\nLoading dataset: ", data_dir)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    class_names = train_ds.class_names

    return class_names, train_ds, val_ds


def configure_for_performance(ds):
    return ds.cache() \
        .shuffle(buffer_size=1000) \
        .prefetch(buffer_size=AUTOTUNE)
    # .batch(BATCH_SIZE) \


def train(classes, train_ds, val_ds, epochs=4):
    print("\nTrain the model ....")

    # https://www.tensorflow.org/tutorials/images/classification
    data_augmentation = tf.keras.Sequential([
        # layers.experimental.preprocessing.RandomFlip("horizontal", seed=SEED),  # already done in input image dataset
        # layers.experimental.preprocessing.RandomFlip("vertical", seed=SEED),  # already done in input image dataset
        layers.experimental.preprocessing.RandomRotation(0.05),
        layers.experimental.preprocessing.RandomZoom(0.05),
        tf.keras.layers.experimental.preprocessing.RandomContrast(
            0.05, seed=SEED),
        # tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1, seed=SEED)
    ])

    # 90%  -> epochs=16 / 32 / 32 / MaxPooling2D           / flatten            / 128 / 29
    # 91%  -> epochs=16 / 32 / 32 / MaxPooling2D           / flatten            / 256 / 29
    # 85%  -> epochs=16 / 32 / 32 / MaxPooling2D           / flatten            / 256 / 29 + RandomRotation
    # 89%  -> epochs=16 / 32 / 32 / MaxPooling2D           / flatten            / 256 / 29 + RandomContrast
    # 94%  -> epochs=16 / 32 / 32 / MaxPooling2D           / flatten            / 256 / 29 + RandomZoom
    # 92%  -> epochs=16 / 32 / 32 / MaxPooling2D           / flatten            / 256 / 29 + DropOut
    # 94%  -> epochs=20 / 32 / 32 / MaxPooling2D           / flatten            / 256 / 29 + DropOut
    # 73%  -> epochs=20 / 32 / 32 / MaxPooling2D           / flatten            / 256 / 29 + all augmentations
    # 84%  -> epochs=32 / 32 / 32 / MaxPooling2D           / flatten            / 256 / 29 + all augmentations
    # 94%  -> epochs=72 / 32 / 32 / MaxPooling2D           / flatten            / 256 / 29 + all augmentations
    model = tf.keras.Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),  # dropout
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(len(classes))
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # Create a callback that saves the model's weights
    # checkpoint_dir = os.path.dirname(checkpointPath)
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,
    #                                                  save_weights_only=True,
    #                                                  save_freq=10 * BATCH_SIZE,
    #                                                  verbose=1)
    # Save the weights using the `checkpoint_path` format
    # model.save_weights(checkpointPath.format(epoch=0))

    training = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        # callbacks=[cp_callback],  # Pass callback to training
        verbose=1
    )

    model.save(MODEL_PATH, overwrite=True, include_optimizer=True)

    model.summary()

    return model, training


def displayTraining(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def _train():
    epochs = 72
    (class_names, train_ds, val_ds) = load(DATASET_ROOT_DIR + 'train-2')
    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    model, training = train(class_names, train_ds, val_ds, epochs)
    displayTraining(training, epochs)


_train()
