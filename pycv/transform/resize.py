import numpy as np
from pycv._lib.core_support import resize_py

__all__ = [
    'bilinear_resize',
    'nearest_neighbour_resize',
]


########################################################################################################################

def bilinear_resize(
        inputs: np.ndarray,
        height: int,
        width: int,
        axis: tuple | None = None
) -> np.ndarray:
    return resize_py.resize_2d(inputs, height, width, 'bilinear', axis)


def nearest_neighbour_resize(
        inputs: np.ndarray,
        height: int,
        width: int,
        axis: tuple | None = None
) -> np.ndarray:
    return resize_py.resize_2d(inputs, height, width, 'nn', axis)
