import numpy as np
from pycv.colors._utils import color_dispatcher
from pycv.colors._coordinate import *

__all__ = [
    'rgb2gray',
    'rgb2yuv',
    'yuv2rgb',
    'rgb2hsv',
]

########################################################################################################################

@color_dispatcher(n_channels=3, same_type=True)
def rgb2gray(
        image: np.ndarray
) -> np.ndarray:
    """
    Convert RGB img to grayscale.

    Parameters
    ----------
    image : numpy.ndarray
        RGB img with shape (..., RGB).

    Returns
    -------
    gray_image : numpy.ndarray
        Grayscale img with the same dtype as the input.

    """
    h = np.array(RGB2GRAY, image.dtype)

    return image @ h


@color_dispatcher(n_channels=3, same_type=True)
def rgb2yuv(
        image: np.ndarray
) -> np.ndarray:
    """
    Convert RGB img to YUV.

    Parameters
    ----------
    image : numpy.ndarray
        Input RGB img array. It should have size of 3 on the last dimension (shape[..., RGB]).

    Returns
    -------
    yuv_image : numpy.ndarray
        YUV img array with the same dtype and shape as the input (shape[..., YUV]).

    Raises
    ------
    ValueError
        If the input array doesn't have the expected number of channels.
    """

    h = np.array(RGB2YUV, image.dtype)
    image = (image @ h) + np.array([0, 0.5, 0.5], image.dtype)

    return image


@color_dispatcher(n_channels=3, same_type=True)
def yuv2rgb(
        image: np.ndarray
) -> np.ndarray:
    """
    Convert YUV img to RGB.

    Parameters
    ----------
    image : numpy.ndarray
        Input YUV img array. It should have size of 3 on the last dimension (shape[..., YUV]).

    Returns
    -------
    RGB_image : numpy.ndarray
        RGB img array with the same dtype and shape as the input (shape[..., RGB]).

    Raises
    ------
    ValueError
        If the input array doesn't have the expected number of channels.
    """
    h = np.array(YUV2RGB, image.dtype)
    image = (image - np.array([0, 0.5, 0.5], image.dtype)) @ h
    return image


@color_dispatcher(n_channels=3, same_type=False)
def rgb2hsv(
        image: np.ndarray
) -> np.ndarray:
    """
    Convert RGB img to HSV.

    Parameters
    ----------
    image : numpy.ndarray
        Input RGB img array. It should have size of 3 on the last dimension (shape[..., RGB]).

    Returns
    -------
    HSV_image : numpy.ndarray
        HSV img array with the same dtype and shape as the input (shape[..., HSV]).

    Raises
    ------
    ValueError
        If the input array doesn't have the expected number of channels.
    """

    out = np.zeros_like(image)

    v = np.amax(image, axis=-1)

    ignor_settings = np.seterr(invalid='ignore')

    delta = image.ptp(-1)
    s = delta / v
    s[delta == 0] = 0.

    red_cond = image[..., 0] == v
    green_cond = image[..., 1] == v
    blue_cond = image[..., -1] == v

    h = np.zeros_like(v)
    h[red_cond] = (image[red_cond, 1] - image[red_cond, 2]) / delta[red_cond]
    h[green_cond] = 2. + (image[green_cond, 2] - image[green_cond, 0]) / delta[green_cond]
    h[blue_cond] = 4. + (image[blue_cond, 0] - image[blue_cond, 1]) / delta[blue_cond]

    h = (h * 6.) % 1
    h[delta == 0] = 0.

    np.seterr(**ignor_settings)

    out[..., 0] = h
    out[..., 1] = s
    out[..., 2] = v

    out[np.isnan(out)] = 0.

    return out

########################################################################################################################
