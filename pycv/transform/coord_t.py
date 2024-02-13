import numpy as np
from pycv._lib._src_py import pycv_transform

__all__ = [
    'resize',
    'rotate'
]


########################################################################################################################

def resize(
        inputs: np.ndarray,
        output_shape: tuple,
        order: int = 1,
        anti_alias_filter: bool | None = None,
        sigma: float | None = None,
        padding_mode: str = 'constant',
        constant_value: float | int | None = 0,
        preserve_dtype: bool = False
) -> np.ndarray:
    """
    order:
        0 = nearest neighbor
        1 = linear
        2 = quadratic
        3 = cubic
    """
    return pycv_transform.resize(inputs, output_shape, order, anti_alias_filter, sigma, padding_mode, constant_value, preserve_dtype)

########################################################################################################################


def rotate(
        inputs: np.ndarray,
        angle: float,
        order: int = 1,
        axis: tuple | None = None,
        reshape: bool = True,
        padding_mode: str = 'constant',
        constant_value: float | int | None = 0,
        preserve_dtype: bool = False
) -> np.ndarray:
    """
    order:
        0 = nearest neighbor
        1 = linear
        2 = quadratic
        3 = cubic
    """
    return pycv_transform.rotate(inputs, angle, order, axis, reshape, padding_mode, constant_value, preserve_dtype)

########################################################################################################################

