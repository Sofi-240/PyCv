import numpy as np
import math
from ..array_api.dtypes import get_dtype_info
import numbers

__all__ = [
    'valid_offset',
    'cast_kernel_dilation',
    'border_mask',
    'reshape_1d_kernel',
    'color_mapping_range',
    'gen_binary_table',
]


########################################################################################################################

def valid_offset(
        kernel_shape: tuple,
        offset: tuple | None
) -> tuple:
    if offset is None:
        if not all((s % 2) != 0 for s in kernel_shape):
            raise ValueError(f'If kernel dimensions length is even offset must be given')
        return tuple(s // 2 for s in kernel_shape)
    if len(kernel_shape) != len(offset):
        raise ValueError(
            f'Number of dimensions in kernel and offset does not match: {len(kernel_shape)} != {len(offset)}'
        )
    if any(o < 0 or o >= s for o, s in zip(offset, kernel_shape)):
        raise ValueError('offset is out of kernel bounds')
    return offset


def cast_kernel_dilation(
        kernel: np.ndarray,
        dilation: tuple | list | int = 1
) -> np.ndarray:
    """
    Generate a dilated kernel.

    Parameters
    ----------
    kernel : numpy.ndarray
        Original convolution kernel.
    dilation : tuple of int, list of int, or int, optional
        Dilation values for each dimension. If int, the same dilation is applied to all dimensions.

    Returns
    -------
    dilated_kernel : numpy.ndarray
        Dilated kernel.

    Raises
    ------
    TypeError:
        If kernel is not an instance of numpy.ndarray.
    ValueError:
        If the number of dimensions in dilation does not match the kernel's number of dimensions.
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError('Kernel must be an instance of numpy.ndarray')

    if isinstance(dilation, numbers.Number):
        if dilation == 1:
            return kernel.copy()
        dilation = (int(dilation),) * kernel.ndim

    if len(dilation) != kernel.ndim:
        raise ValueError(
            f'Number of dimensions in dilation and kernel do not match: {len(dilation)} != {kernel.ndim}'
        )

    if all(d == 1 for d in dilation):
        return kernel.copy()

    casted_kernel = np.zeros(
        tuple(k + ((d - 1) * k) - 1 if d != 1 else k for k, d in zip(kernel.shape, dilation)), dtype=kernel.dtype
    )
    casted_kernel[tuple(slice(None, None, d) for d in dilation)] = kernel

    return casted_kernel


def border_mask(
        image_shape: tuple | list,
        kernel_shape: tuple | list,
        center: tuple | list | None = None
) -> np.ndarray:
    """
    Create a boolean mask with img shape that is 1 on the border and 0 everywhere else.

    Parameters
    ----------
    image_shape : tuple or list of int
        Shape of the img.
    kernel_shape : tuple or list of int
        Shape of the kernel.
    center : tuple or list of int or None, optional
        Tuple of indices to the center of the kernel. If not provided, it is assumed to be the center of the kernel.

    Returns
    -------
    mask : np.ndarray of bool
        Array with img shape that is 1 on the border and 0 everywhere else.

    Raises
    ------
    ValueError:
        If the number of dimensions in the kernel or center does not match the number of dimensions in the img.
    """
    if len(image_shape) != len(kernel_shape):
        raise ValueError(
            f'Number of dimensions in kernel and img does not match: {len(kernel_shape)} != {len(image_shape)}'
        )
    if center is None:
        center = tuple(s // 2 for s in kernel_shape)

    if len(center) != len(kernel_shape):
        raise ValueError(
            f'Number of dimensions in kernel and center does not match: {len(kernel_shape)} != {len(center)}'
        )

    border_weight = tuple((c, int(s - c - 1)) for s, c in zip(kernel_shape, center))

    mask = np.zeros(image_shape, dtype=bool)
    for i, (b1, b2) in enumerate(border_weight):
        np.moveaxis(mask, i, 0)[:b1] = 1
        np.moveaxis(mask, i, 0)[-b2:] = 1

    return mask


def reshape_1d_kernel(
        kernel: np.ndarray,
        ndim: int,
        filter_dim: int
) -> np.ndarray:
    """
    Reshape a 1D kernel to the specified number of dimensions.

    Parameters
    ----------
    kernel : np.ndarray
        Input 1D kernel array.
    ndim : int
        Number of dimensions for the reshaped kernel.
    filter_dim : int
        The dimension along which the kernel will be reshaped.

    Returns
    -------
    reshaped_kernel : np.ndarray
        Reshaped kernel array.

    Raises
    ------
    TypeError:
        If the input kernel is not of type numpy.ndarray.
    ValueError:
        If filter_dim is greater than or equal to ndim.
    ValueError:
        If the reshaped kernel has more than one dimension larger than 1.
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError(f'Kernel must be of type numpy.ndarray, got {type(kernel)}')

    _N = max(kernel.shape)

    if filter_dim >= ndim:
        raise ValueError(f'filter_dim must be smaller than ndim')

    new_shape = [1] * ndim
    new_shape[filter_dim] = _N

    try:
        reshaped_kernel = kernel.reshape(new_shape)
    except ValueError:
        raise ValueError('Kernel has more than one dimension larger than 1')

    return reshaped_kernel


########################################################################################################################


def default_binary_strel(
        ndim: int,
        connectivity: int = 1,
        hole: bool = False
) -> np.ndarray:
    """
    Generate a binary structuring element.

    Parameters
    ----------
    ndim : int
        Number of dimensions for the structuring element.
    connectivity : int, optional
        Connectivity value specifying the neighborhood relationship. Must be in the range from 1 (no diagonal
        elements are neighbors) to ndim (all elements are neighbors).
    hole : bool, optional
        If True, create a structuring element with a hole in the center.

    Returns
    -------
    strel : np.ndarray of bool
        Binary structuring element.

    Raises
    ------
    ValueError:
        If the connectivity value is not in the valid range.
    """
    if connectivity < 1 or connectivity > ndim:
        raise ValueError(
            f'Connectivity value must be in the range from 1 (no diagonal elements are neighbors) '
            f'to ndim (all elements are neighbors)'
        )
    ax = np.arange(0, 3)
    cords = np.meshgrid(*(ax,) * ndim)
    cords_cond = np.abs(cords[0] - 1)
    for c in cords[1:]:
        cords_cond += np.abs(c - 1)
    out = np.array(cords_cond <= connectivity, dtype=bool)
    if hole:
        out[(1,) * ndim] = 0

    return out


def default_cc_strel(
        ndim: int,
        connectivity: int = 1,
        hole: bool = True
) -> np.ndarray:
    strel = default_binary_strel(ndim, connectivity, hole)

    cords = np.ravel_multi_index(np.indices(strel.shape), strel.shape)
    mid = cords[(1,) * ndim]

    return np.where(cords > mid, False, strel)


########################################################################################################################


def color_mapping_range(
        values: np.ndarray,
        method: str,
        mod_value: int = 16
) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        values = np.asarray(values)

    dt = get_dtype_info(values.dtype)
    if dt.kind == 'f':
        raise TypeError(f'values dtype must be int or uint got float dtype')

    min_, max_ = np.min(values), np.max(values)

    if min_ < 0 or max_ > 255:
        raise ValueError('values need to be in range of 0 - 255')

    if method == 'sqr':
        ranges = list(math.isqrt(i) for i in range(256))
    elif method == 'log':
        ranges = [0] + list(int(math.log2(i)) for i in range(1, 256))
    elif method == 'linear':
        ranges = [0] + list(1 + int(i // mod_value) for i in range(1, 256))
    else:
        raise ValueError(f'method need to be "sqr", "log" or "linear"')

    return np.array(ranges, dtype=np.uint8)


########################################################################################################################


def gen_binary_table(rank: int) -> np.ndarray:
    change_point = [0] * rank
    change_point[-1] = 1

    for i in range(rank - 2, -1, -1):
        change_point[i] = change_point[i + 1] << 1

    size = change_point[0] << 1

    binary = np.zeros((rank * size,), np.uint8)

    for i in range(rank):
        val = 0
        counter = change_point[i]
        p = i
        for j in range(size):
            binary[p] = val
            counter -= 1
            if counter == 0:
                counter = change_point[i]
                val = abs(val - 1)
            p += rank

    binary = np.reshape(binary, (-1, rank))
    return binary
