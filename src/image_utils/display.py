"""
Image rendering utility module
"""
import matplotlib.pyplot as plt
import cv2


def convert2Plot(image):
    """
    Convert cv2 BGR images to RGB images to render images in true colors

    :param image: input BGR image
    :return: output RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# -------------------------------------------------------
# Display two images
def display2(a, b, title1="Original", title2="Edited"):
    """
    Display 2 images side by side

    :param a:
    :param b:
    :param title1:
    :param title2:
    :return:
    Refs:
        https://stackoverflow.com/questions/50630825/matplotlib-imshow-distorting-colors
    """
    plt.subplot(211), plt.imshow(convert2Plot(a)), plt.title(title1)
    plt.xticks([]), plt.yticks([])

    plt.subplot(212), plt.imshow(convert2Plot(b)), plt.title(title2)
    plt.xticks([]), plt.yticks([])

    plt.show()


def display3(a, b, c, titlea="first", titleb="second", titlec="third"):
    """
    Display 3 images on a single row

    :param a: first image
    :param b: second image
    :param c: third image
    :param titlea: title for first image
    :param titleb: title for second image
    :param titlec: title for third image
    """
    plt.subplot(311), plt.imshow(convert2Plot(a)), plt.title(titlea)
    plt.xticks([]), plt.yticks([])

    plt.subplot(312), plt.imshow(convert2Plot(b)), plt.title(titleb)
    plt.xticks([]), plt.yticks([])

    plt.subplot(313), plt.imshow(convert2Plot(c)), plt.title(titlec)
    plt.xticks([]), plt.yticks([])

    plt.show()


def display_one(a, title="Original"):
    """
    Display a single image
    :param a:  image
    :param title:  image title
    """
    plt.imshow(convert2Plot(a)), plt.title(title)
    plt.show()
