import numpy as np
import numbers
from pycv._lib.array_api.dtypes import cast, get_dtype_info
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.filters_support.filters import c_convolve
from pycv._lib.filters_support.utils import default_axis
from pycv._lib.filters_support.windows import edge_kernel
from pycv._lib.filters_support.morphology import c_gray_ero_or_dil
from pycv._lib.filters_support.kernel_utils import border_mask

__all__ = [
    'edge_filters',
    'filter_with_convolve',
    'PUBLIC'
]

PUBLIC = []


########################################################################################################################

def if_pad_is_same_or_constant(
        image: np.ndarray,
        kernel_shape: tuple,
        padding_mode: str
):
    if padding_mode not in ['same', 'constant']:
        return image
    mask = border_mask(image.shape, kernel_shape)
    return c_gray_ero_or_dil(0, image, mask=mask)


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
        offset: tuple | None = None,
        padding_mode: str = 'valid',
        **pad_kw
) -> np.ndarray:
    image = np_compliance(image)
    dtype = image.dtype

    image = cast(image, np.float64)

    if axis is None:
        axis = default_axis(image.ndim, min(image.ndim, 2))
    elif isinstance(axis, numbers.Number):
        axis = (axis,)
    if any(ax >= image.ndim for ax in axis):
        raise ValueError('axis is out of range for array dimensions')

    magnitude = len(axis) > 1

    kernel = edge_kernel(smooth_values, edge_values, image.ndim, axis[0])
    kernel_shape = kernel.shape

    output = c_convolve(image, kernel, offset=offset, padding=padding_mode, **pad_kw)

    if magnitude:
        output *= output
        for ax in axis[1:]:
            kernel = edge_kernel(smooth_values, edge_values, image.ndim, ax)
            tmp = c_convolve(image, kernel, padding=padding_mode, **pad_kw)
            output += (tmp * tmp)
        output = np.sqrt(output) / np.sqrt(image.ndim)

    if_pad_is_same_or_constant(output, kernel_shape, padding_mode=padding_mode)

    if preserve_dtype:
        output = cast(output, dtype)

    return output

########################################################################################################################


def filter_with_convolve(
        image: np.ndarray,
        kernel: np.ndarray,
        output: np.ndarray | None = None,
        axis: int | None = None,
        preserve_dtype: bool = False,
        offset: int | tuple | None = None,
        padding_mode: str = 'valid',
        **pad_kw
) -> np.ndarray:
    image = np_compliance(image)
    dtype = get_dtype_info(image.dtype)
    image = cast(image, np.float64)

    if get_dtype_info(kernel.dtype).kind != 'f':
        raise ValueError('kernel dtype need to be float')

    if kernel.ndim == 1:
        if axis is None:
            axis = image.ndim - 1

        elif not isinstance(axis, numbers.Number):
            raise ValueError('axis size is larger then image dimensions')

        if axis >= image.ndim:
            raise ValueError('axis is out of range for array dimensions')

    ret = c_convolve(image, kernel, output, axis, padding_mode=padding_mode, flip=True, offset=offset, **pad_kw)

    if preserve_dtype:
        return cast(ret, dtype.type) if ret is not None else cast(output, dtype.type)

    return output

########################################################################################################################















