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
    if isinstance(threshold, np.ndarray):
        if threshold.shape != image.shape:
            raise ValueError('threshold shape and image shape need to be equal')
    return np.where(image > threshold, True, False)


def im_threshold(
        image: np.ndarray, threshold: str | Thresholds, *args, **kwargs
) -> np.ndarray | tuple[np.ndarray, int | float | np.ndarray]:
    if isinstance(threshold, str):
        threshold = Thresholds[threshold.upper()]
    if not isinstance(threshold, Thresholds):
        raise ValueError(f'{threshold} need to be type of str or Thresholds member')
    th = threshold(image, *args, **kwargs)
    return im_binarize(image, th), th

########################################################################################################################
