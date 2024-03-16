import numpy as np
from .._lib.filters_support.thresholding import Thresholds

__all__ = [
    'otsu_threshold',
    'kapur_threshold',
    'li_and_lee_threshold',
    'minimum_threshold',
    'minimum_error_threshold',
    'mean_threshold',
    'adaptive_threshold',
    'Thresholds',
    'im_binarize',
    'im_threshold'
]


########################################################################################################################


otsu_threshold = Thresholds.OTSU
kapur_threshold = Thresholds.KAPUR
li_and_lee_threshold = Thresholds.LI_AND_LEE
minimum_error_threshold = Thresholds.MINIMUM_ERROR
mean_threshold = Thresholds.MEAN
minimum_threshold = Thresholds.MINIMUM
adaptive_threshold = Thresholds.ADAPTIVE


########################################################################################################################

def im_binarize(image: np.ndarray, threshold: int | float | np.ndarray) -> np.ndarray:
    """
    Binarize the input image based on the given threshold.

    Parameters:
        image (np.ndarray): Input image.
        threshold (int | float | np.ndarray): Threshold value or array for binarization. If threshold is an array, it must
            have the same shape as the input image.

    Returns:
        np.ndarray: Binarized image where pixels greater than the threshold are True and others are False.

    """
    if isinstance(threshold, np.ndarray):
        if threshold.shape != image.shape:
            raise ValueError('threshold shape and image shape need to be equal')
    return np.where(image > threshold, True, False)


def im_threshold(
        image: np.ndarray, threshold: str | Thresholds, *args, **kwargs
) -> np.ndarray | tuple[np.ndarray, int | float | np.ndarray]:
    """
    Apply thresholding to the input image.

    Parameters:
        image (np.ndarray): Input image.
        threshold (str | Thresholds): Thresholding method to use. It can be either a string representing the name of the
            thresholding method or a member of the Thresholds enum.
        *args: Additional positional arguments to be passed to the thresholding function.
        **kwargs: Additional keyword arguments to be passed to the thresholding function.

    Returns:
        np.ndarray | tuple[np.ndarray, int | float | np.ndarray]: If only the binarized image is returned, it's a
        numpy array representing the binarized version of the input image. If both the binarized image and the threshold
        value are returned, it's a tuple containing the binarized image and the computed threshold value.

    """
    if isinstance(threshold, str):
        threshold = Thresholds[threshold.upper()]
    if not isinstance(threshold, Thresholds):
        raise ValueError(f'{threshold} need to be type of str or Thresholds member')
    th = threshold(image, *args, **kwargs)
    return im_binarize(image, th), th

########################################################################################################################
