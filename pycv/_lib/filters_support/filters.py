import numpy as np
import typing
import numbers
from pycv._lib.array_api.dtypes import cast
from pycv._lib._inspect import isfunction
from pycv._lib.array_api.array_pad import get_padding_width, pad
from pycv._lib.array_api.regulator import check_finite
from pycv._lib.array_api.shapes import output_shape, atleast_nd
from pycv._lib.filters_support.kernel_utils import cast_kernel_dilation
from pycv._lib.decorator import registrate_decorator
from pycv._lib.core import ops

FLIPPER = (1, 0, 2)

__all__ = [
    'valid_kernels',
    'filter_dispatcher',
    'apply_filter',
    'apply_filter_function',
    'PUBLIC'
]
PUBLIC = []

DILATION = 1
PADDING_MODE = 'valid'
STRIDE = 1
FLIP = True


########################################################################################################################

def valid_kernels(
        kernel: np.ndarray,
        array_rank: int,
        flip: bool = FLIP,
        dilation: int | tuple = DILATION
) -> tuple[np.ndarray, tuple]:
    if not check_finite(kernel):
        raise ValueError('Kernel must not contain infs or NaNs')
    filter_dim = kernel.ndim
    if filter_dim > 3:
        raise ValueError(f'Convolution for 4D or above is not supported, got kernel with rank of {filter_dim}')
    if flip:
        kernel = np.flip(kernel, FLIPPER[:filter_dim]) if filter_dim > 1 else np.flip(kernel, 0)
    kernel = cast_kernel_dilation(kernel, dilation)
    if filter_dim > 1:
        kernel = atleast_nd(kernel, array_rank, raise_err=False, expand_pos=0)
    return kernel, kernel.shape


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


@registrate_decorator(kw_syntax=True)
def filter_dispatcher(func, *args, **kwargs):
    axis = kwargs.get('axis', None)
    dilation = kwargs.get('dilation', DILATION)
    flip = kwargs.get('flip', FLIP)
    padding_mode = kwargs.get('padding_mode', PADDING_MODE)
    offset = kwargs.get('offset', None)
    stride = kwargs.get('stride', STRIDE)

    if len(args) != 3:
        raise ValueError('args must contain (inputs, kernels, outputs)')
    inputs, kernel, output = args[0], args[1], args[2]

    input_output = output is not None
    filters_as_size = isinstance(kernel, (tuple, numbers.Number))

    array_rank = inputs.ndim

    if not check_finite(inputs):
        raise ValueError('Inputs must not contain infs or NaNs')

    if input_output and not check_finite(output):
        raise ValueError('Output must not contain infs or NaNs')

    if filters_as_size:
        kernel_shape = kernel
        if isinstance(kernel, numbers.Number):
            kernel_shape = (kernel,)
        if not all(isinstance(n, numbers.Number) for n in kernel):
            raise ValueError('Kernel size need to be tuple of ints')
    else:
        kernel, kernel_shape = valid_kernels(kernel, array_rank, flip, dilation)
        kernel = cast(kernel, inputs.dtype)

    filter_dim = len(kernel_shape)
    if axis is None:
        axis = tuple(i for i in range(filter_dim))
    elif isinstance(axis, numbers.Number):
        if filter_dim > 1:
            raise ValueError('axis size not match to kernel dimensions')
        axis = (axis,)

    if not all(ax < array_rank for ax in axis):
        raise ValueError('axis is out of range for array dimensions')

    if len(tuple(set(axis))) != len(axis):
        raise ValueError("axis need to be unique")

    axis_bool = [False] * array_rank
    for ax in axis:
        axis_bool[ax] = True

    if filter_dim == 1:
        sz = kernel_shape[0]
        kernel_shape = tuple(sz if ax else 1 for ax in axis_bool)

    if offset is None:
        offset = tuple(s // 2 for s in kernel_shape)
    else:
        if filter_dim != len(offset):
            raise ValueError(f'Number of dimensions in kernel and offset does not match: {filter_dim} != {len(offset)}')
        if any((o < 0 or o >= ks) for o, ks in zip(offset, kernel_shape)):
            raise ValueError('Invalid offset')

    if padding_mode != 'valid':
        pad_width = get_padding_width(kernel_shape, offset, flip=False, image_shape=inputs.shape)
        inputs = pad(inputs, pad_width, mode=padding_mode, **kwargs)

    if not all(na > nk for na, nk in zip(inputs.shape, kernel_shape)):
        raise ValueError("Kernel dimensions cannot be larger than the input array's dimensions.")

    if not all((nk - 1) >= 0 for nk in kernel_shape):
        raise ValueError("Kernel shape is too small.")

    outputs_shape = output_shape(inputs.shape, kernel_shape, stride)
    output = get_output(output, inputs, outputs_shape)

    if np.all(inputs == 0):
        output[(None,) * array_rank] = 0.
        return None if input_output else output

    if not filters_as_size:
        if filter_dim == 1:
            kernel = np.reshape(kernel, kernel_shape)
        ops.convolve(inputs, kernel, output, offset)
        return None if input_output else output
    func(inputs, kernel_shape, output, *args[3:], **kwargs)
    return None if input_output else output


########################################################################################################################

@filter_dispatcher
def apply_filter(
        inputs: np.ndarray,
        kernel: np.ndarray,
        output: np.ndarray,
        axis: int | tuple | None = None,
        stride: int | tuple | list = 1,
        dilation: int | tuple | list = 1,
        padding_mode: str = 'valid',
        flip: bool = True,
        **padkw
) -> np.ndarray | None:
    pass


@filter_dispatcher
def apply_rank_filter(
        inputs: np.ndarray,
        kernel_size: tuple,
        output: np.ndarray,
        rank: int,
        footprint: np.ndarray | None,
        axis: int | tuple | None = None,
        stride: int | tuple | list = 1,
        dilation: int | tuple | list = 1,
        padding_mode: str = 'valid',
        **padkw
) -> np.ndarray | None:
    if footprint is None:
        footprint = np.ones(kernel_size, bool)
    elif not isinstance(footprint, np.ndarray):
        raise TypeError('footprint need to be type of numpy.ndarray')
    else:
        if footprint.dtype != bool:
            raise ValueError('footprint dtype need to be bool')
        if footprint.shape != kernel_size:
            raise ValueError('footprint not equal to kernel_size')

    ops.rank_filter(inputs, footprint, output, rank, None)


@filter_dispatcher
def apply_filter_function(
        inputs: np.ndarray,
        kernel_size: tuple,
        output: np.ndarray,
        function: typing.Callable,
        axis: int | tuple | None = None,
        stride: int | tuple | list = 1,
        dilation: int | tuple | list = 1,
        padding_mode: str = 'valid',
        **padkw
) -> np.ndarray | None:
    pass

########################################################################################################################
