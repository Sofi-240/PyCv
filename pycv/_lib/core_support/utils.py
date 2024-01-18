import numpy as np
from typing import Iterable, Any
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.array_api.shapes import atleast_nd
from pycv._lib.filters_support.kernel_utils import valid_offset, cast_kernel_dilation

__all__ = [
    'ctype_border_mode',
    'as_sequence',
    'fix_kernel_shape',
    'get_output',
    'get_kernel',
    'valid_kernel_shape_with_ref',
    'valid_same_shape',
    'PUBLIC'
]
PUBLIC = []


########################################################################################################################

def ctype_border_mode(
        mode: str
) -> int:
    if mode == 'valid':
        return 2
    elif mode == 'reflect':
        return 3
    elif mode == 'constant':
        return 4
    elif mode == 'symmetric':
        return 5
    elif mode == 'wrap':
        return 6
    elif mode == 'edge':
        return 7
    else:
        raise RuntimeError('border mode not supported')


########################################################################################################################

def as_sequence(
        sequence: Any,
        rank: int
) -> tuple:
    if not isinstance(sequence, str) and isinstance(sequence, Iterable):
        sequence = tuple(sequence)
    else:
        sequence = (sequence,) * rank
    if len(sequence) != rank:
        raise RuntimeError("sequence argument must have length equal to input rank")
    return sequence


def fix_kernel_shape(shape: int, axis: tuple, nd: int) -> tuple:
    if nd < len(axis):
        raise RuntimeError('n dimensions is smaller then axis size')
    axis_bool = [False] * nd
    for ax in axis:
        if ax >= nd:
            raise RuntimeError('axis is out of range for array dimensions')
        axis_bool[ax] = True
    kernel_shape = tuple(shape if ax else 1 for ax in axis_bool)
    return kernel_shape


########################################################################################################################

def get_output(
        output: np.ndarray | type | np.dtype | None,
        inputs: np.ndarray,
        shape: tuple | None = None
) -> tuple[np.ndarray, bool]:
    if shape is None:
        shape = inputs.shape
    if output is None:
        output = np.zeros(shape, dtype=inputs.dtype)
    elif isinstance(output, (type, np.dtype)):
        output = np.zeros(shape, dtype=output)
    elif not isinstance(output, np.ndarray):
        raise ValueError("output can be np.ndarray or type instance")
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    share_memory = np.may_share_memory(inputs, output)
    return output, share_memory


def get_kernel(
        kernel: np.ndarray,
        array_nd: int,
        dilation: int | tuple | None = None,
        offset: int | tuple | None = None,
        axis: int | None = None
) -> tuple[np.ndarray, tuple]:
    kernel = np_compliance(kernel, arg_name='Kernel', _check_finite=True)

    filter_dim = kernel.ndim
    if dilation is not None:
        kernel = cast_kernel_dilation(kernel, dilation)

    if offset is not None:
        offset = as_sequence(offset, filter_dim)
    offset = valid_offset(kernel.shape, offset)

    if array_nd != filter_dim and filter_dim == 1:
        if axis is None:
            axis = array_nd - 1

        kernel_shape = fix_kernel_shape(kernel.size, (axis, ), array_nd)
        kernel = np.reshape(kernel, kernel_shape)
        offset_scalar, offset = offset[0], tuple()
        for i in range(array_nd):
            offset += (0,) if i != axis else (offset_scalar,)
    elif array_nd != filter_dim:
        kernel = atleast_nd(kernel, array_nd, raise_err=False, expand_pos=0)
        offset = (0,) * (array_nd - filter_dim) + offset

    if not kernel.flags.contiguous:
        kernel = kernel.copy()

    return kernel, offset


########################################################################################################################

def valid_kernel_shape_with_ref(kernel_shape: tuple, image_shape: tuple):
    err1 = "Kernel shape cannot be negative."
    err2 = "Kernel shape cannot be zero."
    err3 = "Kernel dimensions cannot be larger than the input array's dimensions."

    for na, nk in zip(image_shape, kernel_shape):
        if nk < 0:
            raise RuntimeError(err1)
        if nk == 0:
            raise RuntimeError(err2)
        if nk > na:
            raise RuntimeError(err3)


def valid_same_shape(*arrays: np.ndarray) -> bool:
    if len(arrays) == 0: return True
    shape = arrays[0].shape
    for arr in arrays[1:]:
        if shape != arr.shape:
            return False
    return True
