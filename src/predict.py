"""
Model loading and prediction execution
"""
import glob
import pathlib
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from image_utils import utils as ut

from tensorflow.keras.preprocessing import image

DATASET_ROOT_DIR = "../resources/images/"
MODEL_DIR = "model"
MODEL_PATH = MODEL_DIR + "/resistors-model.h5"
IMG_HEIGHT = 176
IMG_WIDTH = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
"""
Category of resistors corresponding to the prediction values
Prediction output values is an integer between 0 and 28
"""
CLASSES = ['10', '100', '100k', '10k', '150', '1M', '1k',
           '20', '200', '20k', '220', '220k', '270', '2k', '2k2',
           '300k', '330', '3k3',
           '470', '470k', '47k', '4k7',
           '510', '51k', '5k1',
           '680', '680k', '68k', '6k8']


def preprocess(filename, height, width):
    """
    Apply preprocessing on image before prediction to improve accuracy

    :param filename: file path to load
    :param height: target height
    :param width: target width
    :return: improved image
    """
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cvt_image = ut.shadow_remove(cvt_image)  # best accuracy with shadow removal
    # cvt_image = _gaussianBlur(cvt_image) # best accuracy without blur
    cvt_image = cv2.resize(cvt_image, (width, height))
    return Image.fromarray(cvt_image)  # convert to PIL image


def loadImageArray(filename):
    """
    Load an image from disk and return an image array to feed Keras

    :param filename: image's file path
    :return: image array
    """
    # pil_img = image.load_img(filename, target_size=(IMG_HEIGHT, IMG_WIDTH)) # PIL image
    img = preprocess(filename, height=IMG_HEIGHT, width=IMG_WIDTH)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def restoreModel(verbose=False):
    """
    Load the model from disk

    :param verbose: True -> display model summary (architecture, layers, ...)
    :return: model
    """
    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model(MODEL_PATH)
    if verbose:
        # Show the model architecture
        model.summary()
    return model


def evaluateModel(model, test_ds):
    """
    Evaluate the model accuracy based on a test dataset

    :param model: model
    :param test_ds: test dataset
    :return: model
    """
    # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
    #     1./255)
    # normalized_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    # images, labels = next(iter(normalized_ds))

    images, labels = next(iter(test_ds))

    loss, acc = model.evaluate(images, labels, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    return model


def restore_and_predict():
    """
    Restore the model from disc and execute predictions based on images stored in a directory
    """
    model = restoreModel(False)

    # evaluateModel(model, data)

    failed_images = []
    failed = 0
    count = 0
    # check prediction each image from 'raw' dataset
    for name in glob.glob(DATASET_ROOT_DIR + 'raw/**/*.png'):
        path = pathlib.PurePath(name)
        value = path.parent.name
        count += 1

        img = loadImageArray(name)
        predictions = model.predict(img)

        score = tf.nn.softmax(predictions[0])
        if value != CLASSES[np.argmax(score)]:
            failed_images.append(name)
            failed += 1

        if count % 200 == 0:
            print("{:>4d} - {:.2f} %".format(count, 100 * ((count - failed) / count)))

    print("Performances: failed:{} / ok: {} / total: {} ok -> {:.2f} %".format(
        failed,
        count - failed,
        count,
        100 * ((count - failed) / count))
    )


restore_and_predict()
