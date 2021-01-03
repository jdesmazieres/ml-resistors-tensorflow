"""" Training image dataset root directory """
TRAIN_DATASET_DIR = "../resources/images/preprocessed"
"""" Training image dataset root directory """
TEST_DATASET_DIR = "../resources/images/raw"
""" Test image dataset path filter """
TEST_DATASET_PATH = TEST_DATASET_DIR + "/**/*.png"
""" Model save directory """
MODEL_DIR = "../model"
""" Model save file path """
MODEL_PATH = MODEL_DIR + "/resistors-model.h5"
""" Image height """
IMG_HEIGHT = 176
""" Image width """
IMG_WIDTH = 64
""" Loading image batch size """
BATCH_SIZE = 32
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
