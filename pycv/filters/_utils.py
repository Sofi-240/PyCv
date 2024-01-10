import numpy as np
import numbers
from pycv._lib.array_api.dtypes import cast
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.filters_support.kernel_utils import border_mask
from pycv._lib.filters_support.filters import apply_filter
from pycv._lib.filters_support.windows import edge_kernel

__all__ = [
    'edge_filters',
    'PUBLIC'
]

PUBLIC = []


########################################################################################################################

def if_pad_is_same_or_constant(
        image: np.ndarray,
        kernel_shape: tuple,
        padding_mode: str
):
    # if padding_mode not in ['same', 'constant']:
    #     return image
    # mask = border_mask(image.shape, kernel_shape)
    return


def kernel_size_valid(
        kernel_size: int | tuple,
        axis: tuple,
        ndim: int
) -> tuple:
    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * len(axis)

    if len(kernel_size) != len(axis):
        raise ValueError('Kernel size and axis size dont match')

    min_ax, max_ax = min(axis), max(axis)
    valid_size = [1] * (ndim - min_ax)

    for a, k in zip(axis, kernel_size):
        valid_size[a - min_ax] = k

    return tuple(valid_size)


########################################################################################################################

def edge_filters(
        image: np.ndarray,
        smooth_values: np.ndarray,
        edge_values: np.ndarray,
        axis: int | tuple | None = None,
        preserve_dtype: bool = False,
        as_float: bool = True,
        padding_mode: str = 'valid',
        **pad_kw
) -> np.ndarray:
    image = np_compliance(image)
    dtype = image.dtype

    if as_float:
        image = cast(image, np.float64)

    if axis is None:
        axis = (image.ndim - 1, image.ndim - 2)
    else:
        if isinstance(axis, numbers.Number):
            axis = (axis,)

    magnitude = len(axis) > 1

    kernel = edge_kernel(smooth_values, edge_values, image.ndim, axis[0])
    kernel_shape = kernel.shape
    output = apply_filter(image, kernel, None, padding=padding_mode, **pad_kw)

    if magnitude:
        output *= output
        for ax in axis[1:]:
            kernel = edge_kernel(smooth_values, edge_values, image.ndim, ax)
            tmp = apply_filter(image, kernel, None, padding=padding_mode, **pad_kw)
            output += (tmp * tmp)
        output = np.sqrt(output) / np.sqrt(image.ndim)

    if_pad_is_same_or_constant(output, kernel_shape, padding_mode=padding_mode)

    if preserve_dtype:
        output = cast(output, dtype)

    return output

########################################################################################################################
