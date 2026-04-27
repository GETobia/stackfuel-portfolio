import os
import re

import numpy as np
import pandas as pd
from PIL import Image

ROWS = 28
COLUMNS = 28

_BLACK_UPPER_BOUND = 15
_WHITE_LOWER_BOUND = 240

_cat_columns = ["letter", "number", "augmentation"]


def _filename_data(filename):
    """
    Split filename of image from the braille dataset into its relevant parts.
    Args:
        filename (str): Expects to match r'[a-z]{1}1[.]JPG[0-9]{1,2}[a-z]{3}[.]jpg' like 'a1.JPG3rot.jpg'

    Returns:
        List of letter, number, augmentation
    """
    letter1, desc = filename.split(".")[:-1]
    num, aug = re.split(r"([0-9]+)", desc)[1:]
    return [letter1[0], num, aug]


def _greyscale_pixels(filepath):
    """
    List greyscale pixel data of ROWSxCOLUMNS image.
    Args:
        filepath (str): Location of the image.

    Returns:
        List of greyscale pixel values.
    """
    img = Image.open(filepath).convert("L")
    return list(np.asarray(img.getdata(), dtype=np.uint8))


def braille_image_data(filepath):
    """Combine _greyscale_pixels and _filename_data"""
    filename = filepath.split("/")[-1]
    return [*_greyscale_pixels(filepath), *_filename_data(filename)]


def braille_images_dataframe(folderpath):
    """Create dataframe for all files in folderpath"""
    filenames = sorted(os.listdir(folderpath))
    columns = pd.Index(
        [*[i for i in range(ROWS * COLUMNS)], *_cat_columns],
        dtype="object",
    )
    data = [
        braille_image_data(os.path.join(folderpath, filename)) for filename in filenames
    ]

    return pd.DataFrame(data, columns=columns)


def seriestoimage(series):
    """
    Pillow Image corresponding to reshaped array from pandas Series.
    Args:
        series (pandas Series object, uint8): Must have its numerical entries in first ROWS*COLUMNS indices.
    Returns:
        Pillow Image made out of the reshaped series values, ROWS rows and COLUMNS columns.
    """
    return Image.fromarray(
        np.array(series[: ROWS * COLUMNS], dtype="uint8").reshape(ROWS, COLUMNS)
    )


def read_cleaned(filepath):
    """
    Read and clean processed braille csv file. Convert numbers to uint8 and objects to category.
    Numeric columns correspond to the first ROWS*COLUMNS indices.
    Returns:
        Pandas Data Frame with the correct data types.
    """
    df = pd.read_csv(filepath)

    dtypes = {
        **{str(i): "uint8" for i in range(ROWS * COLUMNS)},
        **{cat: "category" for cat in _cat_columns},
    }
    return df.astype(dtypes)
