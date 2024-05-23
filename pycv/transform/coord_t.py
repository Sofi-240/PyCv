import numpy as np
from .._lib._src_py import pycv_transform
from .._lib._src_py._geometric_transform import (ProjectiveTransform, RidgeTransform, SimilarityTransform, AffineTransform)

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
        preserve_dtype: bool = True
) -> np.ndarray:
    """
    Resize the input array to the specified output shape.

    Parameters:
        inputs (numpy.ndarray): The input array to be resized.
        output_shape (tuple): The desired output shape.
        order (int, optional): The order of interpolation (0=nearest, 1=linear, 2=quadratic, 3=cubic). Defaults to 1.
        anti_alias_filter (bool or None, optional): Whether to apply an anti-aliasing filter when downsampling. Defaults to None.
        sigma (float or None, optional): The standard deviation for the Gaussian filter if anti_alias_filter is True. Defaults to None.
        padding_mode (str, optional): The padding mode to use for elements outside the boundaries of the input array. Defaults to 'constant'.
        constant_value (float or int or None, optional): The constant value to use for padding if padding_mode is 'constant'. Defaults to 0.
        preserve_dtype (bool, optional): Whether to preserve the data type of the input array in the output. Defaults to False.

    Returns:
        numpy.ndarray: The resized array.

    Notes:
        - If `preserve_dtype` is False, the output array will have dtype float64.

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
    Rotate the input array by the specified angle.

    Parameters:
        inputs (numpy.ndarray): The input array to be rotated.
        angle (float): The angle of rotation in degrees.
        order (int, optional): The order of interpolation (0=nearest, 1=linear, 2=quadratic, 3=cubic). Defaults to 1.
        axis (tuple or None, optional): The axis of rotation. If None, the rotation is applied to the plane of the first two axes. Defaults to None.
        reshape (bool, optional): Whether to reshape the output array to fit the rotated image. Defaults to True.
        padding_mode (str, optional): The padding mode to use for elements outside the boundaries of the input array. Defaults to 'constant'.
        constant_value (float or int or None, optional): The constant value to use for padding if padding_mode is 'constant'. Defaults to 0.
        preserve_dtype (bool, optional): Whether to preserve the data type of the input array in the output. Defaults to False.

    Returns:
        numpy.ndarray: The rotated array.

    Notes:
        - If `preserve_dtype` is False, the output array will have dtype float64.
        - If `reshape` is False, the output array will have the same shape as the input array, but parts of the rotated image may be cropped.
        - If `axis` is provided, it should be a tuple specifying the rotation axis.

    """
    return pycv_transform.rotate(inputs, angle, order, axis, reshape, padding_mode, constant_value, preserve_dtype)


########################################################################################################################

def geometric_transform(
        inputs: np.ndarray,
        transform_matrix: ProjectiveTransform | np.ndarray,
        order: int = 1,
        axis: tuple | None = None,
        output_shape: tuple | None = None,
        padding_mode: str = 'constant',
        constant_value: float | int | None = 0,
        preserve_dtype: bool = True
) -> np.ndarray:
    """
        Apply a geometric transformation to the input array.

        Parameters:
            inputs (numpy.ndarray): The input array to be transformed.
            transform_matrix (numpy.ndarray): The transformation matrix.
            order (int, optional): The order of interpolation (0=nearest, 1=linear, 2=quadratic, 3=cubic). Defaults to 1.
            axis (tuple or None, optional): The axes along which the transformation is applied. Defaults to None.
            output_shape (tuple or None, optional): Output shape. Defaults to None.
            padding_mode (str, optional): The padding mode to use for elements outside the boundaries of the input array. Defaults to 'constant'.
            constant_value (float or int or None, optional): The constant value to use for padding if padding_mode is 'constant'. Defaults to 0.
            preserve_dtype (bool, optional): Whether to preserve the data type of the input array in the output. Defaults to False.

        Returns:
            numpy.ndarray: The transformed array.

        Notes:
            - If `preserve_dtype` is False, the output array will have dtype float64.
        """
    return pycv_transform.geometric_transform(inputs, transform_matrix, order, axis, output_shape, padding_mode, constant_value,
                                              preserve_dtype)

########################################################################################################################
