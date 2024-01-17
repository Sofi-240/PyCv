import numpy as np
import math
from pycv._lib.array_api.dtypes import get_dtype_info
import numbers

__all__ = [
    'valid_offset',
    'cast_kernel_dilation',
    'unraveled_offsets',
    'ravel_offsets',
    'unravel_offsets',
    'isvalid_neighbor_raveled',
    'isvalid_neighbor_unraveled',
    'border_mask',
    'reshape_1d_kernel',
    'color_mapping_range',
    'PUBLIC'
]

PUBLIC = []


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


def unraveled_offsets(
        kernel: np.ndarray,
        center: tuple | list | None = None
) -> np.ndarray:
    """
    Generate neighborhood offsets in unraveled coordinate space.

    Parameters
    ----------
    kernel : np.ndarray
        The kernel where zero values are not included in the neighborhood. The array can have any dtype.
    center : tuple of int, or None, optional
        Tuple of indices to the center of the structuring element. If not provided, it is assumed to be the center
        of the structuring element.

    Returns
    -------
    unraveled_neighborhood : np.ndarray
        The offsets to a sample's neighbors in the unraveled form (number of offsets, kernel dimensions).

    Raises
    ------
    TypeError:
        If the kernel is not a numpy.ndarray instance.
    ValueError:
        If the number of dimensions in the center does not match the number of dimensions in the kernel.
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError(f'Kernel must be an instance of numpy.ndarray, got {type(kernel)}')
    if center is None:
        center = tuple(s // 2 for s in kernel.shape)
    if len(center) != kernel.ndim:
        raise ValueError(f'Number of dimensions in center and kernel do not match: {len(center)} != {kernel.ndim}')
    unraveled_neighborhood = np.stack([(idx - c) for idx, c in zip(np.nonzero(kernel), center)], axis=-1)
    return unraveled_neighborhood


def ravel_offsets(
        offsets_unraveled: np.ndarray | list,
        image_shape: tuple,
        order: str = 'C'
) -> np.ndarray:
    """
    Generate neighborhood offsets in raveled coordinate space.

    Parameters
    ----------
    offsets_unraveled : np.ndarray or list[list]
        Neighborhood offsets in unraveled coordinate space. Shape: (number of offsets, dimensions).
    image_shape : tuple of int
        Tuple representing the shape of the img.
    order : str, optional
        The index order. 'C' (default) means to index the elements in row-major, 'F' means to index the elements
        in column-major.

    Returns
    -------
    raveled_neighborhood : np.ndarray
        The offsets to a sample's neighbors in the raveled form.

    Raises
    ------
    ValueError:
        If the number of dimensions in offsets_unraveled does not match the number of dimensions in the img.
    ValueError:
        If order is not 'C' or 'F'.
    """
    if not isinstance(offsets_unraveled, np.ndarray):
        offsets_unraveled = np.asarray(offsets_unraveled, dtype=np.int64)

    if offsets_unraveled.shape[-1] != len(image_shape):
        raise ValueError(
            f'Number of dimensions in offsets_unraveled and img do not match:'
            f' {offsets_unraveled.shape[-1]} != {len(image_shape)}'
        )

    if order == 'F':
        offsets_unraveled = offsets_unraveled[:, ::-1]
        image_shape = image_shape[::-1]
    elif order != 'C':
        raise ValueError("Order must be 'C' or 'F'")

    jump = image_shape[1:] + (1,)
    jump = np.cumprod(jump[::-1])[::-1]
    raveled_neighborhood = (offsets_unraveled * jump).sum(axis=1)

    return raveled_neighborhood


def unravel_offsets(
        offsets: np.ndarray | list[int],
        center: tuple,
        image_shape: tuple,
        order: str = 'C'
) -> np.ndarray:
    """
    Return offsets in unraveled coordinate space.

    Parameters
    ----------
    offsets : np.ndarray or list of int
        The offsets in raveled coordinate space.
    center : tuple of int
        Tuple of indices to the center coordinate.
    image_shape : tuple of int
        Tuple representing the shape of the img.
    order : str, optional
        The index order. 'C' (default) means to index the elements in row-major, 'F' means to index the elements
        in column-major.

    Returns
    -------
    unraveled_neighborhood : np.ndarray
        The offsets coordinate in the unraveled form (number of offsets, kernel dimensions).

    Raises
    ------
    ValueError:
        If the number of dimensions in the center does not match the number of dimensions in the img.
    ValueError:
        If order is not 'C' or 'F'.
    """
    if len(center) != len(image_shape):
        raise ValueError(f'Number of dimensions in center and img do not match: {len(center)} != {len(image_shape)}')
    if order != 'C' and order != 'F':
        raise ValueError("Order must be 'C' or 'F'")
    Np = len(offsets)
    Nd = len(image_shape)
    shift = np.ravel_multi_index(center, image_shape, order=order)
    unraveled_neighborhood = np.zeros((Np, Nd), dtype=np.int64)

    for i, cord in enumerate(offsets):
        c = np.unravel_index(cord + shift, image_shape, order=order)
        for d in range(Nd):
            unraveled_neighborhood[i, d] = c[d] - center[d]

    return unraveled_neighborhood


def isvalid_neighbor_raveled(
        raveled_index: int,
        neighbor_offset: tuple | list,
        image_shape: tuple | list,
        order: str = 'C'
) -> bool:
    """
    Check whether a neighbor of a given index is inside the img.

    Parameters
    ----------
    raveled_index : int
        The index in raveled coordinate space.
    neighbor_offset : tuple or list of int
        Tuple of offsets from the index in each dimension.
    image_shape : tuple or list of int
        Tuple representing the shape of the img.
    order : str, optional
        The index order. 'C' (default) means to index the elements in row-major, 'F' means to index the elements
        in column-major.

    Returns
    -------
    isvalid : bool
        True if valid, else False.

    Raises
    ------
    ValueError:
        If the number of dimensions in the neighbor and img does not match.
    ValueError:
        If order is not 'C' or 'F'.
    """
    if len(neighbor_offset) != len(image_shape):
        raise ValueError(
            f'Number of dimensions in img and neighbor does not match: {len(image_shape)} != {len(neighbor_offset)}'
        )
    if order != 'C' and order != 'F':
        raise ValueError("Order must be 'C' or 'F'")
    unraveled_index = np.unravel_index(raveled_index, image_shape, order=order)
    return all(0 <= i + n < s for i, n, s in zip(unraveled_index, neighbor_offset, image_shape))


def isvalid_neighbor_unraveled(
        unraveled_index: tuple | list,
        neighbor_offset: tuple | list,
        image_shape: tuple | list,
) -> bool:
    """
    Check whether a neighbor of a given index is inside the img.

    Parameters
    ----------
    unraveled_index : tuple or list of int
        The index in unraveled coordinate space.
    neighbor_offset : tuple or list of int
        Tuple of offsets from the index in each dimension.
    image_shape : tuple or list of int
        Tuple representing the shape of the img.

    Returns
    -------
    isvalid : bool
        True if valid, else False.

    Raises
    ------
    ValueError:
        If the number of dimensions in the neighbor and img does not match.
    """
    if len(neighbor_offset) != len(image_shape):
        raise ValueError(
            f'Number of dimensions in img and neighbor does not match: {len(image_shape)} != {len(neighbor_offset)}'
        )
    return all(0 <= i + n < s for i, n, s in zip(unraveled_index, neighbor_offset, image_shape))


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
        ranges = list(int(i // mod_value) for i in range(256))
    else:
        raise ValueError(f'method need to be "sqr", "log" or "linear"')

    return np.array(ranges, dtype=np.uint8)

########################################################################################################################