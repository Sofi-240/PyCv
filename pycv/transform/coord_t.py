import numpy as np
from pycv._lib.core_support import interpolation_py as interp

__all__ = [
    'resize',
]


########################################################################################################################

def resize(
        inputs: np.ndarray,
        output_shape: tuple,
        order: int = 1,
        padding_mode: str = 'constant',
        constant_value: float | int | None = 0
) -> np.ndarray:
    """
    order:
        0 = nearest neighbor
        1 = linear
        2 = quadratic
        3 = cubic
    """
    return interp.resize(inputs, output_shape, order, padding_mode, constant_value)

########################################################################################################################


def rotate(
        inputs: np.ndarray,
        angle: float,
        order: int = 3,
        axis: tuple | None = None,
        reshape: bool = True,
        padding_mode: str = 'constant',
        constant_value: float | int | None = 0
) -> np.ndarray:
    """
    order:
        0 = nearest neighbor
        1 = linear
        2 = quadratic
        3 = cubic
    """
    return interp.rotate(inputs, angle, order, axis, reshape, padding_mode, constant_value)

########################################################################################################################

