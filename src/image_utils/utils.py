"""
Image processing utility module

This module provides high-level, dedicated methods on top of openCV module
"""
import os
import cv2
import numpy as np


def contrast(images, clip=1.0, grid=8):
    """
    Increase the image contrast
    :param images: list of input images
    :param clip: clip limit
    :param grid: tile grid size
    :return: list of processed images
    """
    contrasted = []
    for i in range(len(images)):
        img = images[i]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        contrasted.append(final)
    return contrasted


def filePaths(rootpath):
    """
    Recursively retrieve the path of each image from the root directory
    :param rootpath: images root directory
    :return: list of path (str)
    """
    return sorted([os.path.join(rootpath, file)
                   for file in os.listdir(rootpath)
                   if file.endswith('.png')])


# --------------------------------
# resize a liste of images, and return the list of resized images
def resize(images, width, height, verbose=False):
    """
    resize a list of images, and return the list of resized images
    :param images: input list of images
    :param width: target width
    :param height: target height
    :param verbose: verbose True|False
    :return: list of resized images
    """
    dim = (width, height)
    res_img = []
    for i in range(len(images)):
        res = cv2.resize(images[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)
        if verbose:
            try:
                print('Original size: ', images[i].shape, ' - Altered size: ', res_img[i].shape)
            except AttributeError:
                print("shape not found")

    return res_img


def gaussianBlur(images, bsize=5):
    """
    Apply gaussian blur to the input image list

    This is used to reduce noise
    :param images: input list of images
    :param bsize:
    :return: processed image list
    """
    processed = []
    for i in range(len(images)):
        _p = cv2.GaussianBlur(images[i], (bsize, bsize), 0)
        processed.append(_p)
    return processed


# ----------------------------------
# Remove noise
# Using Gaussian Blur
def blur(images, bsize=5):
    """
    Apply blur to the input image list

    This is used to reduce noise
    :param images: input list of images
    :param bsize:
    :return: processed image list
    """
    processed = []
    for i in range(len(images)):
        _p = cv2.blur(images[i], (bsize, bsize))
        processed.append(_p)
    return processed


def medianBlur(images, bsize=5):
    """
    Apply median blur to the input image list

    This is used to reduce noise
    :param images: input list of images
    :param bsize:
    :return: processed image list
    """
    processed = []
    for i in range(len(images)):
        _p = cv2.medianBlur(images[i], bsize)
        processed.append(_p)
    return processed


# ----------------------------------
# Remove noise
# Using Median Blur
def bilateralFilter(images, d=5, sigmaColor=1, sigmaSpace=1):
    """
    Apply bilateral filtering to the input image list

    This is used to reduce noise
    :param images: input list of images
    :param d:
    :param sigmaColor:
    :param sigmaSpace:
    :return: processed image list
    """
    processed = []
    for i in range(len(images)):
        _p = cv2.bilateralFilter(images[i], d, sigmaColor, sigmaSpace)
        processed.append(_p)
    return processed


# https://medium.com/arnekt-ai/shadow-removal-with-open-cv-71e030eadaf5
def shadow_remove(images):
    """
    Apply shadow removal to the input image list

    This is used to improve image quality
    :param images: input list of images
    :return: processed image list
    """
    processed = []
    for i in range(len(images)):
        img = images[i]
        rgb_planes = cv2.split(img)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((5, 5), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 51)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)
        shadowremov = cv2.merge(result_norm_planes)
        processed.append(shadowremov)
    return processed


def image_resize(images, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Not ratio altering resizing
    :param images: input list of images
    :param width: target width (height is inferred)
    :param height: target height (width is inferred)
    :param inter: interpolation algorithm
    :return: processed image list
    """
    processed = []
    for i in range(len(images)):
        image = images[i]
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)
        print("Original: ", image.shape, ", resized: ", resized.shape)
        processed.append(resized)

    # return the resized image
    return processed
