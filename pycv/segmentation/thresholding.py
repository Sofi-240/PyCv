import numpy as np
from typing import Any
from pycv._lib.filters_support.thresholding import Thresholds

__all__ = [
    'otsu_threshold',
    'kapur_threshold',
    'li_and_lee_threshold',
    'minimum_threshold',
    'minimum_error_threshold',
    'mean_threshold',
    'adaptive_threshold',
    'im_binarize',
    'im_threshold'
]


########################################################################################################################

def otsu_threshold(
        image: np.ndarray,
) -> int | float:
    return Thresholds.OTSU(image)


def kapur_threshold(
        image: np.ndarray,
) -> int | float:
    return Thresholds.KAPUR(image)


def li_and_lee_threshold(
        image: np.ndarray,
) -> int | float:
    return Thresholds.LI_AND_LEE(image)


def minimum_error_threshold(
        image: np.ndarray,
) -> int | float:
    return Thresholds.MINIMUM_ERROR(image)


def mean_threshold(
        image: np.ndarray,
) -> int | float:
    return Thresholds.MEAN(image)


def minimum_threshold(
        image: np.ndarray,
        max_iterations: int = 10000
) -> int | float:
    return Thresholds.MINIMUM(image, max_iterations=max_iterations)


def adaptive_threshold(
        image: np.ndarray,
        block_size: tuple | int,
        method: str = 'gaussian',
        method_params: Any = None,
        offset_val: int | float = 0,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
        axis: tuple | None = None
) -> np.ndarray:
    return Thresholds.ADAPTIVE(
        image, block_size, method=method, method_params=method_params, offset_val=offset_val,
        padding_mode=padding_mode, constant_value=constant_value, axis=axis
    )


########################################################################################################################

def im_binarize(
        image: np.ndarray,
        threshold: int | float | np.ndarray,
) -> np.ndarray:
    if isinstance(threshold, np.ndarray):
        if threshold.shape != image.shape:
            raise ValueError('threshold shape and image shape need to be equal')
    return np.where(image > threshold, True, False)


def im_threshold(
        image: np.ndarray,
        threshold: str,
        *args, **kwargs
) -> np.ndarray | tuple[np.ndarray, int | float | np.ndarray]:
    if threshold not in Thresholds:
        raise ValueError(f'{threshold} method is not supported use {Thresholds}')
    th = Thresholds.get_method(threshold)(image, *args, **kwargs)
    return im_binarize(image, th), th

########################################################################################################################
