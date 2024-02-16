import numpy as np
from pycv._lib._src_py.utils import ctype_border_mode, get_output, get_kernel, valid_kernel_shape_with_ref
from pycv._lib._src import c_pycv

__all__ = [
    'convolve',
    'rank_filter',
]


########################################################################################################################

def convolve(
        inputs: np.ndarray,
        kernel: np.ndarray,
        output: np.ndarray | None = None,
        axis: int | None = None,
        stride: int | tuple | None = None,
        dilation: int | tuple | None = None,
        offset: int | tuple | None = None,
        padding_mode: str = 'valid',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    inputs = np.asarray(inputs)

    kernel, offset = get_kernel(kernel, inputs.ndim, dilation=dilation, offset=offset, axis=axis)

    input_shape = inputs.shape
    kernel_shape = kernel.shape

    valid_kernel_shape_with_ref(kernel_shape, inputs.shape)

    if stride is not None:
        raise RuntimeError('stride option is currently not supported')

    if padding_mode == 'valid':
        outputs_shape = tuple(n - k + 1 for n, k in zip(input_shape, kernel_shape))
    else:
        outputs_shape = inputs.shape

    padding_mode = ctype_border_mode(padding_mode)
    output, share_memory = get_output(output, inputs, outputs_shape)
    hold_output = None

    if share_memory:
        hold_output = output
        output, _ = get_output(hold_output.dtype, inputs, outputs_shape)

    if np.all(inputs == 0) and constant_value == 0:
        output[...] = 0.
    else:
        c_pycv.convolve(inputs, kernel, output, offset, padding_mode, constant_value)

    if share_memory:
        hold_output[...] = output
        output = hold_output

    return output


########################################################################################################################

def rank_filter(
        inputs: np.ndarray,
        footprint: np.ndarray,
        rank: int,
        output: np.ndarray | None = None,
        axis: int = None,
        stride: int | tuple | list | None = None,
        offset: tuple | None = None,
        padding_mode: str = 'valid',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    inputs = np.asarray(inputs)

    footprint, offset = get_kernel(footprint, inputs.ndim, offset=offset, axis=axis)

    input_shape = inputs.shape
    footprint_shape = footprint.shape

    valid_kernel_shape_with_ref(footprint_shape, inputs.shape)

    if footprint.dtype != bool:
        footprint = footprint.astype(bool)

    if stride is not None:
        raise RuntimeError('stride option is currently not supported')

    if padding_mode == 'valid':
        outputs_shape = tuple(n - k + 1 for n, k in zip(input_shape, footprint_shape))
    else:
        outputs_shape = inputs.shape

    padding_mode = ctype_border_mode(padding_mode)

    if rank > np.sum(footprint):
        raise ValueError(f'rank is out of range for footprint with n {footprint.size} nonzero elements')

    output, share_memory = get_output(output, inputs, outputs_shape)
    hold_output = None

    if share_memory:
        hold_output = output
        output, _ = get_output(hold_output.dtype, inputs, outputs_shape)

    if np.all(inputs == 0) and constant_value == 0:
        output[...] = 0.
    else:
        c_pycv.rank_filter(inputs, footprint, output, rank, offset, padding_mode, constant_value)

    if share_memory:
        hold_output[...] = output
        output = hold_output

    return output


########################################################################################################################
