import re

import numpy as np
from PIL import Image


def _filename_data(filename):
    """
    Split filename of image from the braille dataset into its relevant parts.
    Args:
        filename (str): Expects to match r'[a-z]{1}1[.]JPG[0-9]{1,2}[a-z]{3}[.]jpg'

    Returns:
        List of letter, number, augmentation
    """
    letter1, desc = filename.split(".")[:-1]
    num, aug = re.split(r"([0-9]+)", desc)[1:]
    return [letter1[0], np.uint8(num), aug]


def _greyscale_pixels(filepath):
    """
    List greyscale pixel data of 28x28 image.
    Args:
        filepath (str): Location of the image.

    Returns:
        List of greyscale pixel values.
    """
    img = Image.open(filepath).convert("L")
    return list(np.asarray(img.getdata(), dtype=np.uint8))


def braille_image_data(filepath):
    """Combine _filename_data and _greyscale_pixels."""
    filename = filepath.split("/")[-1]
    return [*_greyscale_pixels(filepath), *_filename_data(filename)]
