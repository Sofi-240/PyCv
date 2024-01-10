import numpy as np
from numpy.lib.stride_tricks import as_strided
import numbers

__all__ = [
    'atleast_nd',
    'check_nd',
    'flat_to_ndim',
    'sliding_window_view',
    'output_shape',
    'RAVEL_ORDER',
    'PUBLIC'
]

PUBLIC = []

RAVEL_ORDER = 'C'


########################################################################################################################

def atleast_nd(
        inputs: np.ndarray,
        ndim: int,
        raise_err: bool = False,
        expand_pos: int = -1
) -> np.ndarray:
    """
    Expend array dimensions to the given ndim if raise_err is False
    else raise expectation if the dimensions smaller than ndim.

    Parameters
    ----------
    inputs : numpy.ndarray
    ndim: int
    raise_err: bool default False
    expand_pos: inf default -1

    Returns
    -------
    numpy.ndarray.

    Raises
    ------
    ValueError:
        If the dimensions smaller than ndim and raise_err is True.
    """
    if not isinstance(inputs, np.ndarray):
        inputs = np.asarray(inputs)

    if raise_err and inputs.ndim < ndim:
        raise ValueError(
            f'Array need to be with atleast {ndim} dimensions got {inputs.ndim}'
        )

    def _expand(arr: np.ndarray) -> np.ndarray:
        if arr.ndim >= ndim:
            return arr
        arr = np.expand_dims(arr, axis=expand_pos)
        return _expand(arr)

    return _expand(inputs)


def check_nd(
        inputs: np.ndarray,
        ndim: int,
        raise_err: bool = False
) -> bool:
    if not isinstance(inputs, np.ndarray):
        inputs = np.asarray(inputs)
    if inputs.ndim == ndim:
        return True
    if raise_err and inputs.ndim != ndim:
        raise ValueError(
            f'Array need to be with {ndim} dimensions got {inputs.ndim}'
        )
    raise False


def flat_to_ndim(
        inputs: np.ndarray,
        ndim: int,
        order: str = RAVEL_ORDER
) -> np.ndarray:
    """
    Reshape a flattened array to have a specified number of dimensions.

    Parameters
    ----------
    inputs : numpy.ndarray
        N-dimensional array.
    ndim: int
        Number of dimensions for the output array.
    order: str (default 'C')
        ravel order

    Returns
    -------
    outputs : numpy.ndarray
        An array with the specified number of dimensions.

    Raises
    ------
    TypeError:
        If the input is not of type numpy.ndarray.
    ValueError:
        If the array rank is smaller than the specified number of dimensions (ndim).
    """
    if not isinstance(inputs, np.ndarray):
        raise TypeError(f'Input array must be of type numpy.ndarray, got {type(inputs)}')

    arr_shape = inputs.shape
    arr_rank = inputs.ndim
    if arr_rank < ndim:
        raise ValueError(f'Array rank is smaller than ndim.')

    if arr_rank - ndim == 0: return inputs
    product = np.prod(arr_shape[:-ndim + 1])

    return inputs.reshape((product, *arr_shape[-ndim + 1:]), order=order)


def sliding_window_view(
        arr: np.ndarray,
        window_shape: tuple | list,
        stride: int | tuple | list = 1
) -> np.ndarray:
    """
    Generate a sliding window view for the input array.

    Parameters
    ----------
    arr : numpy.ndarray
        N-dimensional array (..., N, M).
    window_shape: tuple or list
        Window dimensions specified by the corresponding shape.
    stride: tuple, list, or int, optional
        Stride values for each dimension. If tuple or list, dimensions must match window dimensions.
        If an int, the same stride is applied to all dimensions.

    Returns
    -------
    outputs : numpy.ndarray
        An N-dimensional array with a shape of (..., W1, W2, ...).

    Raises
    ------
    TypeError:
        If the input is not of type numpy.ndarray.
    ValueError:
        If the window rank exceeds the array rank.
    ValueError:
        If the stride rank is incompatible with the window rank.
    ValueError:
        If the window size is larger than the array in terms of shape.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f'Input array must be of type numpy.ndarray, got {type(arr)}')

    arr_rank = arr.ndim
    arr_shape = arr.shape

    window_rank = len(window_shape)

    if window_rank > arr_rank:
        raise ValueError('Window rank must be smaller or equal to the array rank.')

    dsz = arr_rank - window_rank

    if isinstance(stride, numbers.Number):
        stride = (stride,) * window_rank

    if len(stride) != window_rank:
        raise ValueError('Stride rank is incompatible with window rank.')

    if not all(na > nk for na, nk in zip(arr_shape[-window_rank:], window_shape)):
        raise ValueError("Window dimensions cannot be larger than the input array's dimensions.")

    if not all((nk - 1) >= 0 for nk in window_shape):
        raise ValueError("Window shape is too small.")

    stride = (1,) * dsz + tuple(stride)
    window_shape = (1,) * dsz + tuple(window_shape)

    slices = tuple(slice(None, None, st) for st in stride)

    win_indices_shape = tuple(
        ((arr_s - win_s) // str_s) + 1 for arr_s, win_s, str_s in zip(arr_shape, window_shape, stride))

    new_shape = win_indices_shape + window_shape
    strides = arr[slices].strides + arr.strides

    out = as_strided(arr, shape=new_shape, strides=strides)
    return out


def output_shape(
        input_shape: tuple,
        kernel_shape: tuple,
        stride: int | tuple = 1,
        dilation: int | tuple = 1,
        padding_mode: str = 'valid'
) -> tuple:
    """
    Calculate the output shape after applying a filter with the given parameters.

    Parameters
    ----------
    input_shape : tuple of int
        Represents the array shape before filtering.
    kernel_shape: tuple of int
        Represents the kernel shape.
    stride: tuple or int, optional
        Stride values for each dimension. If tuple, dimensions must match the kernel dimensions.
        If int, the same stride is applied to all dimensions.
    dilation: tuple or int, optional
        Dilation values for each dimension. If tuple, dimensions must match the kernel dimensions.
        If int, the same dilation is applied to all dimensions.
    padding_mode: str, optional
        Padding mode of the array. 'valid' means no padding; otherwise, it is considered padded.

    Returns
    -------
    outputs : tuple of ints
        Represents the output shape.

    Raises
    ------
    ValueError:
        If dilation or stride is a tuple with a size different from the kernel size.
    ValueError:
        If the kernel dimensions' length is not odd.
    """
    if isinstance(dilation, numbers.Number):
        dilation = (int(dilation),) * len(kernel_shape)

    if len(dilation) != len(kernel_shape):
        raise ValueError('Dilation size must be equal to the kernel size.')

    if isinstance(stride, numbers.Number):
        stride = (int(stride),) * len(kernel_shape)

    if len(stride) != len(kernel_shape):
        raise ValueError('Stride size must be equal to the kernel size.')

    if padding_mode == 'valid':
        padding = (0,) * len(kernel_shape)
    else:
        padding = tuple(k // 2 for k in kernel_shape)

    out = tuple(
        np.floor(
            (n + 2 * p - k - (k - 1) * (d - 1)) / s
        ).astype(int) + 1 for n, k, s, d, p in zip(input_shape, kernel_shape, stride, dilation, padding)
    )
    return out


########################################################################################################################
