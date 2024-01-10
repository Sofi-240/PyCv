import numpy as np
import numbers
from pycv._lib.array_api.array_pad import get_padding_width, pad
from pycv._lib.array_api.regulator import check_finite
from pycv._lib.array_api.shapes import output_shape, atleast_nd
from pycv._lib.filters_support.kernel_utils import cast_kernel_dilation, valid_offset
from pycv._lib.core import ops

FLIPPER = (1, 0, 2)

__all__ = [
    'default_axis',
    'fix_kernel_shape',
    'valid_kernels',
    'get_output',
    'c_convolve',
    'c_rank_filter',
    'PUBLIC'
]
PUBLIC = []

DILATION = 1
PADDING_MODE = 'valid'
STRIDE = 1
FLIP = True


########################################################################################################################

def default_axis(nd: int, default_nd: int) -> tuple:
    if default_nd > nd:
        raise ValueError('array dimensions is smaller then the default dimensions')
    axis = tuple(nd - i - 1 for i in range(default_nd))
    return axis


def fix_kernel_shape(shape: int, axis: tuple, nd: int) -> tuple:
    if nd > len(axis):
        raise ValueError('n dimensions is smaller then axis size')
    axis_bool = [False] * nd
    for ax in axis:
        if ax >= nd:
            raise ValueError('axis is out of range for array dimensions')
        axis_bool = True
    kernel_shape = tuple(shape if ax else 1 for ax in axis_bool)
    return kernel_shape


########################################################################################################################

def valid_kernels(
        kernel: np.ndarray,
        array_rank: int,
        flip: bool = FLIP,
        dilation: int | tuple = DILATION,
        offset: int | tuple | None = None,
        filter_dim_bound: int = 3
) -> tuple[np.ndarray, tuple, tuple]:
    if not check_finite(kernel):
        raise ValueError('Kernel must not contain infs or NaNs')
    filter_dim = kernel.ndim
    if filter_dim > filter_dim_bound:
        raise ValueError(f'Convolution for 4D or above is not supported, got kernel with rank of {filter_dim}')
    if flip:
        kernel = np.flip(kernel, FLIPPER[:filter_dim]) if filter_dim > 1 else np.flip(kernel, 0)
    kernel = cast_kernel_dilation(kernel, dilation)
    if filter_dim == 1 and isinstance(offset, numbers.Number):
        offset = (offset,)
    offset = valid_offset(kernel.shape, offset)
    if filter_dim > 1:
        kernel = atleast_nd(kernel, array_rank, raise_err=False, expand_pos=0)
        for _ in range(kernel.ndim - len(offset)):
            offset = (0,) + offset
    return kernel, kernel.shape, offset


def get_output(
        output: np.ndarray | type | np.dtype | None,
        inputs: np.ndarray,
        shape: tuple | None = None
):
    if shape is None:
        shape = inputs.shape
    if output is None:
        output = np.zeros(shape, dtype=inputs.dtype)
    elif isinstance(output, (type, np.dtype)):
        output = np.zeros(shape, dtype=np.dtype)
    elif not isinstance(output, np.ndarray):
        raise ValueError("output can be np.ndarray or type instance")
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    return output


########################################################################################################################

def c_convolve(
        inputs: np.ndarray,
        kernel: np.ndarray,
        output: np.ndarray | None = None,
        axis: int = None,
        stride: int | tuple | list = 1,
        dilation: int | tuple | list = 1,
        padding_mode: str = 'valid',
        flip: bool = True,
        offset: tuple | None = None,
        **padkw
) -> np.ndarray | None:
    nd = inputs.ndim

    if not check_finite(inputs):
        raise ValueError('Inputs must not contain infs or NaNs')

    kernel, kernel_shape, offset = valid_kernels(kernel, nd, flip, dilation, offset)
    k_nd = len(kernel_shape)

    input_output = output is not None

    if k_nd == 1:
        axis = nd - 1 if axis is None else axis
        kernel_shape = tuple(kernel_shape[0] if ax == axis else 1 for ax in range(nd))
        kernel = np.reshape(kernel, kernel_shape)
        if offset is not None:
            offset = tuple(offset[0] if ax == axis else 0 for ax in range(nd))

    if padding_mode != 'valid':
        pad_width = get_padding_width(kernel_shape, offset, flip=False, image_shape=inputs.shape)
        inputs = pad(inputs, pad_width, mode=padding_mode, **padkw)

    if kernel.dtype != inputs.dtype:
        kernel = kernel.astype(inputs.dtype)

    if not all(na > nk for na, nk in zip(inputs.shape, kernel_shape)):
        raise ValueError("Kernel dimensions cannot be larger than the input array's dimensions.")

    if not all((nk - 1) >= 0 for nk in kernel_shape):
        raise ValueError("Kernel shape is too small.")

    outputs_shape = output_shape(inputs.shape, kernel_shape, stride)
    output = get_output(output, inputs, outputs_shape)

    if np.all(inputs == 0):
        output[(None,) * nd] = 0.
        return None if input_output else output

    ops.convolve(inputs, kernel, output, offset)

    return None if input_output else output


def c_rank_filter(
        inputs: np.ndarray,
        footprint: np.ndarray,
        rank: int,
        output: np.ndarray | None = None,
        axis: int = None,
        stride: int | tuple | list = 1,
        padding_mode: str = 'valid',
        offset: tuple | None = None,
        **padkw
) -> np.ndarray | None:
    nd = inputs.ndim

    if not check_finite(inputs):
        raise ValueError('Inputs must not contain infs or NaNs')

    footprint, kernel_shape, offset = valid_kernels(footprint, nd, False, DILATION, offset)
    k_nd = len(kernel_shape)

    if not all(s % 2 != 0 for s in kernel_shape):
        raise ValueError('kernel dimensions size need to be odd')

    input_output = output is not None

    if k_nd == 1:
        axis = nd - 1 if axis is None else axis
        kernel_shape = tuple(kernel_shape[0] if ax == axis else 1 for ax in range(nd))
        footprint = np.reshape(footprint, kernel_shape)
        if offset is not None:
            offset = tuple(offset[0] if ax == axis else 0 for ax in range(nd))

    if padding_mode != 'valid':
        pad_width = get_padding_width(kernel_shape, offset, flip=False, image_shape=inputs.shape)
        inputs = pad(inputs, pad_width, mode=padding_mode, **padkw)

    if footprint.dtype != bool:
        footprint = footprint.astype(bool)

    if not all(na > nk for na, nk in zip(inputs.shape, kernel_shape)):
        raise ValueError("Kernel dimensions cannot be larger than the input array's dimensions.")

    if not all((nk - 1) >= 0 for nk in kernel_shape):
        raise ValueError("Kernel shape is too small.")

    if rank > footprint.size:
        raise ValueError(f'rank is out of range for footprint with size {footprint.size}')

    outputs_shape = output_shape(inputs.shape, kernel_shape, stride)
    output = get_output(output, inputs, outputs_shape)

    if np.all(inputs == 0):
        output[(None,) * nd] = 0.
        return None if input_output else output

    ops.rank_filter(inputs, footprint, output, rank, offset)

    return None if input_output else output


########################################################################################################################



