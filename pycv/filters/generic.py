import numpy as np
import numbers
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.filters_support.windows import gaussian_kernel
from pycv.filters._utils import kernel_size_valid, filter_with_convolve
from pycv._lib.core_support.filters_py import rank_filter
from pycv._lib.filters_support.utils import default_axis


__all__ = [
    'gaussian_filter',
    'mean_filter',
    'median_filter',
    'local_max_filter',
    'local_min_filter',
    'PUBLIC'
]

PUBLIC = [
    'gaussian_filter',
    'mean_filter',
    'median_filter',
    'local_max_filter',
    'local_min_filter',
]


########################################################################################################################


def gaussian_filter(
        image: np.ndarray,
        sigma: float | tuple,
        axis: tuple | None = None,
        preserve_dtype: bool = True,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0
) -> np.ndarray:
    if axis is None:
        axis = default_axis(image.ndim, min(2 if isinstance(sigma, numbers.Number) else 2, image.ndim))
    elif isinstance(axis, numbers.Number):
        axis = (axis,)

    if isinstance(sigma, numbers.Number):
        sigma = (sigma,) * len(axis)

    if len(sigma) != len(axis):
        raise ValueError('Sigma and axis size dont match')

    if len(sigma) == 1 or all(sigma[0] == s for s in sigma[1:]):
        kernel = gaussian_kernel(sigma[0], len(axis))
        valid_shape = kernel_size_valid(kernel.shape[0], axis, len(axis))
        if valid_shape != kernel.shape:
            kernel = np.reshape(kernel, valid_shape)
        output = filter_with_convolve(image, kernel, None, preserve_dtype=preserve_dtype, padding_mode=padding_mode, constant_value=constant_value)
    else:
        output = image.copy()
        for ax, s in zip(axis, sigma):
            kernel = gaussian_kernel(s)
            output = filter_with_convolve(output, kernel, None, axis=ax, preserve_dtype=preserve_dtype,
                                          padding_mode=padding_mode, constant_value=constant_value)

    return output


def mean_filter(
        image: np.ndarray,
        kernel_size: int | tuple,
        axis: int | tuple | None = None,
        preserve_dtype: bool = True,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0
) -> np.ndarray:
    if axis is None:
        axis = default_axis(image.ndim, min(2 if isinstance(kernel_size, numbers.Number) else 2, image.ndim))
    elif isinstance(axis, numbers.Number):
        axis = (axis,)

    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * len(axis)

    kernel_size = kernel_size_valid(kernel_size, axis, image.ndim)

    kernel = np.ones(kernel_size, np.float64) / np.prod(kernel_size, dtype=np.float64)

    return filter_with_convolve(image, kernel, None, preserve_dtype=preserve_dtype, padding_mode=padding_mode, constant_value=constant_value)


def median_filter(
        image: np.ndarray,
        kernel_size: int | tuple,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True)
    if axis is None:
        axis = default_axis(image.ndim, min(2 if isinstance(kernel_size, numbers.Number) else 2, image.ndim))
    elif isinstance(axis, numbers.Number):
        axis = (axis,)

    kernel_size = kernel_size_valid(kernel_size, axis, image.ndim)
    rank = np.prod(kernel_size) // 2

    footprint = np.ones(kernel_size, bool)

    return rank_filter(image, footprint, rank, padding_mode=padding_mode, constant_value=constant_value)


def local_min_filter(
        image: np.ndarray,
        kernel_size: int | tuple,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True)
    if axis is None:
        axis = default_axis(image.ndim, min(2 if isinstance(kernel_size, numbers.Number) else 2, image.ndim))
    elif isinstance(axis, numbers.Number):
        axis = (axis,)

    kernel_size = kernel_size_valid(kernel_size, axis, image.ndim)

    footprint = np.ones(kernel_size, bool)

    return rank_filter(image, footprint, 0, padding_mode=padding_mode, constant_value=constant_value)


def local_max_filter(
        image: np.ndarray,
        kernel_size: int | tuple,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True)
    if axis is None:
        axis = default_axis(image.ndim, min(2 if isinstance(kernel_size, numbers.Number) else 2, image.ndim))
    elif isinstance(axis, numbers.Number):
        axis = (axis,)

    kernel_size = kernel_size_valid(kernel_size, axis, image.ndim)

    footprint = np.ones(kernel_size, bool)

    return rank_filter(image, footprint, np.prod(kernel_size) - 1, padding_mode=padding_mode, constant_value=constant_value)

########################################################################################################################
