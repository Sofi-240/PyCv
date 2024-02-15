import numpy as np
from pycv._lib._src_py import pycv_transform
from pycv._lib._src_py._geometric_transform import (ProjectiveTransform, RidgeTransform,
                                                    SimilarityTransform, AffineTransform)

__all__ = [
    'resize',
    'rotate',
    'ProjectiveTransform',
    'RidgeTransform',
    'SimilarityTransform',
    'AffineTransform',
    'geometric_transform'
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
    return pycv_transform.resize(inputs, output_shape, order, anti_alias_filter, sigma, padding_mode, constant_value,
                                 preserve_dtype)


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

def geometric_transform(
        inputs: np.ndarray,
        transform_matrix: ProjectiveTransform | np.ndarray,
        order: int = 1,
        axis: tuple | None = None,
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
    return pycv_transform.geometric_transform(inputs, transform_matrix, order, axis, padding_mode, constant_value,
                                              preserve_dtype)

########################################################################################################################
