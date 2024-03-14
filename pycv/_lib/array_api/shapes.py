import numpy as np
import numbers

__all__ = [
    'atleast_nd',
    'output_shape',
]


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
