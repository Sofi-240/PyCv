import numpy as np
import numbers
from pycv._lib.array_api.dtypes import cast
from pycv._lib.filters_support.windows import gaussian_kernel
from pycv.filters._utils import kernel_size_valid
from pycv._lib.filters_support.filters import apply_filter, apply_rank_filter

__all__ = [
    'gaussian_filter',
    'mean_filter',
    'median_filter',
    'PUBLIC'
]

PUBLIC = [
    'gaussian_filter',
    'mean_filter',
    'median_filter',
]


########################################################################################################################


def gaussian_filter(
        image: np.ndarray,
        sigma: float | tuple,
        axis: int | tuple | None = None,
        preserve_dtype: bool = True,
        padding_mode: str = 'reflect',
        **pad_kw
) -> np.ndarray:
    dtype = image.dtype
    image = cast(image, np.float64)

    if axis is None:
        axis = (image.ndim - 1, image.ndim - 2)
    else:
        if isinstance(axis, numbers.Number):
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
        output = apply_filter(image, kernel, None, padding_mode=padding_mode, flip=False, **pad_kw)
    else:
        output = image.copy()
        for ax, s in zip(axis, sigma):
            kernel = gaussian_kernel(s)
            output = apply_filter(output, kernel, None, axis=ax, padding_mode=padding_mode, **pad_kw)

    return output if not preserve_dtype else cast(output, dtype)


def mean_filter(
        image: np.ndarray,
        kernel_size: int | tuple,
        axis: int | tuple | None = None,
        preserve_dtype: bool = True,
        padding_mode: str = 'reflect',
        **pad_kw
) -> np.ndarray:
    dtype = image.dtype
    image = cast(image, np.float64)

    if axis is None:
        kn = 1 if isinstance(kernel_size, numbers.Number) else len(kernel_size)
        axis = tuple(image.ndim - (i + 1) for i in range(kn))
    elif isinstance(axis, numbers.Number):
        axis = (axis,)

    kernel_size = kernel_size_valid(kernel_size, axis, image.ndim)

    kernel = np.ones(kernel_size) / np.prod(kernel_size)
    output = apply_filter(image, kernel, None, padding_mode=padding_mode, flip=False, **pad_kw)
    return output if not preserve_dtype else cast(output, dtype)


def median_filter(
        image: np.ndarray,
        kernel_size: int | tuple,
        axis: int | tuple | None = None,
        preserve_dtype: bool = True,
        padding_mode: str = 'reflect',
        **pad_kw
) -> np.ndarray:
    dtype = image.dtype
    image = cast(image, np.float64)

    if axis is None:
        kn = 1 if isinstance(kernel_size, numbers.Number) else len(kernel_size)
        axis = tuple(image.ndim - (i + 1) for i in range(kn))
    elif isinstance(axis, numbers.Number):
        axis = (axis,)

    kernel_size = kernel_size_valid(kernel_size, axis, image.ndim)
    rank = np.prod(kernel_size) // 2
    output = apply_rank_filter(image, kernel_size, None, rank, padding_mode=padding_mode, flip=False, **pad_kw)
    return output if not preserve_dtype else cast(output, dtype)

########################################################################################################################
