import numpy as np
from pycv._lib.array_api.dtypes import cast
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.array_api.array_pad import pad, get_padding_width
from pycv._lib._src_py.utils import valid_axis, fix_kernel_shape
from pycv._lib._src_py import pycv_filters
from pycv._lib.filters_support.windows import gaussian_kernel
from pycv.filters._filters import valid_footprint
from pycv._lib._src_py import pycv_morphology

__all__ = [
    'gaussian_filter',
    'mean_filter',
    'image_filter',
    'median_filter',
    'rank_filter',
    'local_min_filter',
    'local_max_filter'
]


########################################################################################################################

def gaussian_filter(
        image: np.ndarray,
        sigma: float | tuple,
        truncate: float = 3.,
        axis: tuple | None = None,
        preserve_dtype: bool = True,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0
) -> np.ndarray:
    image = np_compliance(image, 'image', _check_finite=True)
    dtype = image.dtype
    image = cast(image, np.float64)

    axis = valid_axis(image.ndim, axis, 2 if np.isscalar(sigma) else len(sigma))

    if np.isscalar(sigma):
        sigma = (sigma,) * len(axis)

    one_pass = len(set(sigma)) == 1
    if one_pass:
        kernel = gaussian_kernel(sigma[0], len(axis), truncate=truncate)

        kernel_shape = fix_kernel_shape(kernel.shape, axis, image.ndim)
        kernel = np.reshape(kernel, kernel_shape)

        output = pycv_filters.convolve(image, kernel, padding_mode=padding_mode, constant_value=constant_value)
    else:
        output = image.copy()
        for s, ax in zip(sigma, axis):
            kernel = gaussian_kernel(s, 1, truncate=truncate)
            output = pycv_filters.convolve(
                output, kernel, axis=ax, padding_mode=padding_mode, constant_value=constant_value
            )

    if preserve_dtype:
        output = cast(output, dtype)

    return output


########################################################################################################################

def mean_filter(
        image: np.ndarray,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        preserve_dtype: bool = True,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0
) -> np.ndarray:
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)
    dtype = image.dtype
    image = cast(image, np.float64)

    footprint = valid_footprint(image.ndim, kernel_size, footprint, axis)

    kernel = footprint.astype(np.float64)
    kernel /= np.sum(kernel)

    output = pycv_filters.convolve(image, kernel, padding_mode=padding_mode, constant_value=constant_value)

    if preserve_dtype:
        output = cast(output, dtype)

    return output


def image_filter(
        image: np.ndarray,
        kernel: np.ndarray,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0
) -> np.ndarray:
    image = np_compliance(image, 'image', _check_finite=True)
    kernel = np_compliance(kernel, 'kernel', _check_finite=True)

    if image.ndim != kernel.ndim and axis is not None:
        axis = valid_axis(image.ndim, axis, kernel.ndim)
        if len(axis) != kernel.ndim:
            raise ValueError('kernel N dimensions dont match with axis length')

        for ax in range(image.ndim):
            if ax not in axis:
                kernel = np.expand_dims(kernel, ax)

    output = pycv_filters.convolve(image, kernel, padding_mode=padding_mode, constant_value=constant_value)

    return output


########################################################################################################################

def median_filter(
        image: np.ndarray,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)

    footprint = valid_footprint(image.ndim, kernel_size, footprint, axis)

    rank = np.sum(footprint) // 2

    output = pycv_filters.rank_filter(image, footprint, rank, padding_mode=padding_mode, constant_value=constant_value)

    return output


def rank_filter(
        image: np.ndarray,
        rank: int,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)

    footprint = valid_footprint(image.ndim, kernel_size, footprint, axis)

    if rank > np.sum(footprint):
        raise ValueError('invalid rank higher then the sum of footprint')

    output = pycv_filters.rank_filter(image, footprint, rank, padding_mode=padding_mode, constant_value=constant_value)

    return output


########################################################################################################################


def local_min_filter(
        image: np.ndarray,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)
    footprint = valid_footprint(image.ndim, kernel_size, footprint, axis)

    if padding_mode not in ['constant', 'valid']:
        image = pad(image, get_padding_width(footprint.shape), mode=padding_mode)
        padding_mode = 'valid'

    output = pycv_morphology.gray_ero_or_dil(0, image, footprint, offset=tuple(s // 2 for s in footprint.shape), border_val=constant_value)

    if padding_mode == 'valid':
        pw = get_padding_width(footprint.shape)
        output = output[tuple(slice(s[0], sh - s[1]) for (s, sh) in zip(pw, image.shape))]

    return output


def local_max_filter(
        image: np.ndarray,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)
    footprint = valid_footprint(image.ndim, kernel_size, footprint, axis)

    if padding_mode not in ['constant', 'valid']:
        image = pad(image, get_padding_width(footprint.shape), mode=padding_mode)
        padding_mode = 'valid'

    output = pycv_morphology.gray_ero_or_dil(1, image, footprint, offset=tuple(s // 2 for s in footprint.shape), border_val=constant_value)

    if padding_mode == 'valid':
        pw = get_padding_width(footprint.shape)
        output = output[tuple(slice(s[0], sh - s[1]) for (s, sh) in zip(pw, image.shape))]

    return output

########################################################################################################################
