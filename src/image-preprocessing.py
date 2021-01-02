""" Raw images preprocessing script

This script allows applying preprocessing on raw images in order to
improve prediction accuracy by normalising training and test images

Refs:
    https://towardsdatascience.com/image-pre-processing-c1aec0be3edf
    https://neptune.ai/blog/data-exploration-for-image-segmentation-and-object-detection
"""
import glob
import cv2
from image_utils import display as dp
from image_utils import utils as ut


def compare(names, img1, img2, label2, count=5):
    for i in range(len(img1[:count])):
        dp.display2(img1[i], img2[i], names[i], label2)


def normalize(fileset: list):
    """
    Normalize input images to improve training and detection accuracy

    The best result is obtained by applying
    1 - shadow remove
    2 - fine gaussian blur

    :param fileset: list of image file paths
    :return: list of image file paths, original image list, normalized image list
    """
    print("\n----------------------------------------------------------------")
    print("List of files in the folder:\n", fileset)

    original = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in fileset]

    # ensure image size
    normalized = ut.resize(original, width=176, height=64)

    # normalized = contrast(normalized)

    # normalized = ut.bilateralFilter(normalized, 9, 25, 25)

    # normalized = ut.medianBlur(normalized, 3)

    normalized = ut.shadow_remove(normalized)

    normalized = ut.gaussianBlur(normalized, 1)

    for i in range(len(fileset)):
        cv2.imwrite(fileset[i], normalized[i])

    return fileset, original, normalized


def main():
    image_path = "../images/resistors/train/"
    # retrieve recursively the image paths from the image's root directory
    files = [f for f in glob.glob(image_path + "**/*.png", recursive=True)]

    # apply normalization on each file (the original file is overwritten)
    (names, original, normalized) = normalize(files)

    compare(files, original, normalized, "Normalized", 9)


main()
