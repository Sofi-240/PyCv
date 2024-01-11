import numpy as np
import typing
from pycv.segmentation._utils import threshold_dispatcher

__all__ = [
    'otsu_threshold',
    'kapur_threshold',
    'li_and_lee_threshold',
    'minimum_threshold',
    'minimum_error_threshold',
    'mean_threshold',
    'adaptive_threshold',
    'PUBLIC'
]

PUBLIC = [
    'otsu_threshold',
    'kapur_threshold',
    'li_and_lee_threshold',
    'minimum_threshold',
    'minimum_error_threshold',
    'mean_threshold',
    'adaptive_threshold',
]


########################################################################################################################

@threshold_dispatcher(method_name='otsu')
def otsu_threshold(
        image: np.ndarray,
) -> int | float:
    pass


@threshold_dispatcher(method_name='kapur')
def kapur_threshold(
        image: np.ndarray,
) -> int | float:
    pass


@threshold_dispatcher(method_name='li_and_lee')
def li_and_lee_threshold(
        image: np.ndarray,
) -> int | float:
    pass


@threshold_dispatcher(method_name='minimum_error')
def minimum_error_threshold(
        image: np.ndarray,
) -> int | float:
    pass


@threshold_dispatcher(method_name='mean')
def mean_threshold(
        image: np.ndarray,
) -> int | float:
    pass


@threshold_dispatcher(method_name='minimum')
def minimum_threshold(
        image: np.ndarray,
        max_iterations: int = 10000
) -> int | float:
    pass


@threshold_dispatcher(method_name='adaptive')
def adaptive_threshold(
        image: np.ndarray,
        kernel_size: tuple | int,
        method: str = 'gaussian',
        method_params: typing.Any = None,
        offset_val: int | float = 0,
        padding_mode: str = 'reflect',
        **pad_kw
) -> np.ndarray:
    pass

########################################################################################################################
