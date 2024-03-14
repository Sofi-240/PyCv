import numpy as np
from .._lib._decorator import wrapper_decorator
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

def _threshold(func):
    name = '_'.join(func.__name__.split('_')[:-1]).upper()
    if name not in Thresholds:
        raise ValueError(f'{name} is not member of Thresholds')

    def _wrapper(f, *args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_decorator(wrapper=_wrapper)(Thresholds[name].function)


########################################################################################################################

@_threshold
def otsu_threshold(image: np.ndarray, nbin: int | None = None) -> int | float:
    pass


@_threshold
def kapur_threshold(image: np.ndarray, nbin: int | None = None) -> int | float:
    pass


@_threshold
def li_and_lee_threshold(image: np.ndarray, nbin: int | None = None) -> int | float:
    pass


@_threshold
def minimum_error_threshold(image: np.ndarray, nbin: int | None = None) -> int | float:
    pass


@_threshold
def mean_threshold(image: np.ndarray) -> int | float:
    pass


@_threshold
def minimum_threshold(image: np.ndarray, nbin: int | None = None, max_iterations: int = 10000) -> int | float:
    pass


@_threshold
def adaptive_threshold(
        image: np.ndarray,
        block_size: tuple | int,
        method: str = 'gaussian',
        method_params=None,
        offset_val: int | float = 0,
        padding_mode: str = 'reflect',
        constant_value: float = 0,
        axis: tuple | None = None
) -> np.ndarray:
    pass


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
