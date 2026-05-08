import os
import re

import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.filters import gaussian, sato, sobel
from skimage.io import imread
from sklearn.base import BaseEstimator, TransformerMixin

_ROWS = 28
_COLUMNS = 28

_cat_columns = ["letter", "number", "augmentation"]

_accepted_image_extensions = [".jpg", ".jpeg", ".png"]
_filename_regex_pattern = r"^[a-z]{1}1[.]JPG[0-9]{1,2}[a-z]{3}$"


def _filename_data(filename):
    """
    Split filename of image from the braille dataset into its relevant parts.
    Args:
        filename (str): Expects to match the filename regex pattern with an accepted extension, like 'a1.JPG3rot.jpg'

    Returns:
        List of letter, number, augmentation
    """
    letter1, desc = filename.split(".")[:-1]
    num, aug = re.split(r"([0-9]+)", desc)[1:]
    return [letter1[0], num, aug]


def _greyscale_pixels(filepath):
    """
    Greyscale pixel data of ROWSxCOLUMNS image.
    Args:
        filepath (str): Location of the image.

    Returns:
        ndarray of greyscale pixel values.
    """
    img = imread(filepath, as_gray=True)
    return np.reshape(img, _ROWS * _COLUMNS)


def _is_brailleimage_filename(filename):
    """
    Check if filename can be processed into image data.
    Args:
        filename (str): includes .extension.
    Returns:
        Boolean, True iff filename can be processed.
    """
    name, ext = os.path.splitext(filename)
    pattern = re.compile(_filename_regex_pattern)

    return pattern.match(name) and ext in _accepted_image_extensions


def braille_image_data(filepath):
    """Combine _greyscale_pixels and _filename_data."""
    filename = filepath.split("/")[-1]
    return [*_greyscale_pixels(filepath), *_filename_data(filename)]


def braille_images_dataframe(folderpath):
    """Create dataframe for all files in folderpath.
    Args:
        folderpath (str): Path to braille images folder.
    Returns:
        pandas DataFrame with braille image data.
    """
    filenames = sorted(os.listdir(folderpath))
    columns = pd.Index(
        [*[i for i in range(_ROWS * _COLUMNS)], *_cat_columns],
        dtype="object",
    )
    data = [
        braille_image_data(os.path.join(folderpath, filename))
        for filename in filenames
        if _is_brailleimage_filename(filename)
    ]

    return pd.DataFrame(data, columns=columns)


def seriestoimage(series):
    """
    Get ROWSxCOLUMNS matrix numpy array corresponding to reshaped array from pandas Series.
    Args:
        series (pandas Series object): Must have its numerical entries in first ROWS*COLUMNS indices.
    Returns:
        Array made out of the reshaped series values, ROWS rows and COLUMNS columns.
    """
    return np.array(series[: _ROWS * _COLUMNS], dtype=np.float64).reshape(
        _ROWS, _COLUMNS
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
        **{cat: "category" for cat in _cat_columns},
    }
    return df.astype(dtypes)


class HOGTransformer(BaseEstimator, TransformerMixin):
    """HOG transformer for all rowsxcolumns images in DataFrame"""

    rows = _ROWS
    columns = _COLUMNS

    def __init__(
        self,
        y=None,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
        channel_axis=None,
    ):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.visualize = False
        self.transform_sqrt = transform_sqrt
        self.feature_vector = feature_vector
        self.channel_axis = channel_axis

    def _hogify(self, img):
        return hog(
            image=img,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=self.visualize,
            transform_sqrt=self.transform_sqrt,
            feature_vector=self.feature_vector,
            channel_axis=self.channel_axis,
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.array(X)
        arr_imgs = np.array([x.reshape(self.rows, self.columns) for x in X])

        return np.array([self._hogify(x) for x in arr_imgs])


class GaussianTransformer(BaseEstimator, TransformerMixin):
    rows = _ROWS
    columns = _COLUMNS

    def __init__(
        self,
        sigma=1.0,
        mode="nearest",
        cval=0,
        preserve_range=False,
        truncate=4.0,
        channel_axis=None,
        out=None,
    ):
        self.sigma = sigma
        self.mode = mode
        self.cval = cval
        self.preserve_range = preserve_range
        self.truncate = truncate
        self.channel_axis = channel_axis
        self.out = out

    def fit(self, X, y=None):
        return self

    def _gaussify(self, img):
        return gaussian(
            image=img,
            sigma=self.sigma,
            mode=self.mode,
            cval=self.cval,
            preserve_range=self.preserve_range,
            truncate=self.truncate,
            channel_axis=self.channel_axis,
            out=self.out,
        )

    def transform(self, X, y=None):
        X = np.array(X)
        arr_imgs = np.array([x.reshape(self.rows, self.columns) for x in X])

        return np.array([self._gaussify(x).flatten() for x in arr_imgs])


class SobelTransformer(BaseEstimator, TransformerMixin):
    """Sobel-filtered array"""

    rows = _ROWS
    columns = _COLUMNS

    def __init__(self, mask=None, *, axis=None, mode="reflect", cval=0.0):
        self.mask = mask
        self.axis = axis
        self.mode = mode
        self.cval = cval

    def _sobelify(self, img):
        return sobel(
            image=img, mask=self.mask, axis=self.axis, mode=self.mode, cval=self.cval
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.array(X)
        arr_imgs = np.array([x.reshape(self.rows, self.columns) for x in X])

        return np.array([self._sobelify(x).flatten() for x in arr_imgs])


class SatoTransformer(BaseEstimator, TransformerMixin):
    """Sato-filtered array"""

    rows = _ROWS
    columns = _COLUMNS

    def __init__(
        self, sigmas=range(1, 10, 2), black_ridges=True, mode="reflect", cval=0
    ):
        self.sigmas = sigmas
        self.black_ridges = black_ridges
        self.mode = mode
        self.cval = cval

    def _satoify(self, img):
        return sato(
            image=img,
            sigmas=self.sigmas,
            black_ridges=self.black_ridges,
            mode=self.mode,
            cval=self.cval,
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.array(X)
        arr_imgs = np.array([x.reshape(self.rows, self.columns) for x in X])

        return np.array([self._satoify(x).flatten() for x in arr_imgs])
