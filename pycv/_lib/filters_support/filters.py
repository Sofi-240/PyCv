import numpy as np
from pycv._lib.array_api.array_pad import get_padding_width, pad
from pycv._lib.array_api.regulator import check_finite
from pycv._lib.array_api.shapes import output_shape
from pycv._lib.filters_support.utils import valid_kernels, get_output, valid_kernel_shape_with_ref
from pycv._lib.core import ops

FLIPPER = (1, 0, 2)

__all__ = [
    'c_convolve',
    'c_rank_filter',
    'PUBLIC'
]
PUBLIC = []


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

    valid_kernel_shape_with_ref(kernel_shape, inputs.shape)

    outputs_shape = output_shape(inputs.shape, kernel_shape, stride)
    output, share_memory = get_output(output, inputs, outputs_shape)
    hold_output = None

    if share_memory:
        hold_output = output
        output, _ = get_output(hold_output.dtype, inputs, outputs_shape)

    if np.all(inputs == 0):
        output[...] = 0.
    else:
        ops.convolve(inputs, kernel, output, offset)

    if share_memory:
        hold_output[...] = output
        output = hold_output

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

    valid_kernel_shape_with_ref(kernel_shape, inputs.shape)

    if rank > footprint.size:
        raise ValueError(f'rank is out of range for footprint with size {footprint.size}')

    outputs_shape = output_shape(inputs.shape, kernel_shape, stride)
    output, share_memory = get_output(output, inputs, outputs_shape)
    hold_output = None

    if share_memory:
        hold_output = output
        output, _ = get_output(hold_output.dtype, inputs, outputs_shape)

    if np.all(inputs == 0):
        output[...] = 0.
    else:
        ops.rank_filter(inputs, footprint, output, rank, offset)

    if share_memory:
        hold_output[...] = output
        output = hold_output

    return None if input_output else output


########################################################################################################################



