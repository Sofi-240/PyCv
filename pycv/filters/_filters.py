import numpy as np
from .._lib.array_api.dtypes import cast, as_binary_array
from .._lib.array_api.regulator import np_compliance
from .._lib._src_py.utils import valid_axis, fix_kernel_shape
from .._lib._src_py.pycv_filters import convolve
from .._lib.filters_support.windows import edge_kernel
from .._lib._src_py import pycv_morphology
from .._lib.filters_support.kernel_utils import border_mask

__all__ = [
    'edge_filters',
]


########################################################################################################################

def valid_footprint(
        ndim: int,
        kernel_shape: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
) -> np.ndarray:
    if footprint is not None:
        footprint = as_binary_array(footprint, 'footprint')
        kernel_shape = footprint.shape
        if footprint.ndim != ndim and axis is not None:
            axis = valid_axis(ndim, axis, footprint.ndim)
            if len(axis) != footprint.ndim:
                raise ValueError('footprint ndim dont match axis len')
            kernel_shape = fix_kernel_shape(kernel_shape, axis, ndim)
            footprint = np.reshape(footprint, kernel_shape)
    else:
        if kernel_shape is None:
            raise ValueError('one of kernel_shape or footprint must be given')
        elif np.isscalar(kernel_shape):
            kernel_shape = (kernel_shape, )
        axis = valid_axis(ndim, axis, len(kernel_shape))
        kernel_shape = fix_kernel_shape(kernel_shape, axis, ndim)
        footprint = np.ones(kernel_shape, bool)
    return footprint


########################################################################################################################

def edge_filters(
        image: np.ndarray,
        smooth_values: np.ndarray,
        edge_values: np.ndarray,
        offset: tuple | None = None,
        axis: int | tuple | None = None,
        preserve_dtype: bool = False,
        padding_mode: str = 'symmetric',
        constant_value: float | None = 0.0
) -> np.ndarray:
    image = np_compliance(image, 'image', _check_finite=True)
    dtype = image.dtype
    image = cast(image, np.float64)

    axis = valid_axis(image.ndim, axis, 2)

    kernel = edge_kernel(smooth_values, edge_values, image.ndim, axis[0])

    magnitude = len(axis) > 1

    output = convolve(image, kernel, offset=offset, padding_mode=padding_mode, constant_value=constant_value)
    if magnitude:
        output *= output
        for ax in axis[1:]:
            kernel = edge_kernel(smooth_values, edge_values, image.ndim, ax)
            tmp = convolve(image, kernel, offset=offset, padding_mode=padding_mode, constant_value=constant_value)
            output += (tmp * tmp)
        output = np.sqrt(output) / np.sqrt(len(axis))

    if padding_mode == 'constant':
        kernel_shape = tuple(edge_values.size if a in axis else 1 for a in range(image.ndim))
        mask = border_mask(image.shape, kernel_shape)
        output = pycv_morphology.gray_ero_or_dil(0, image, np.ones(kernel_shape), mask=mask)

    if preserve_dtype:
        output = cast(output, dtype)

    return output

########################################################################################################################
