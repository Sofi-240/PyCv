import numpy as np
from typing import Iterable, Any
import numbers
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.array_api.shapes import atleast_nd
from pycv._lib.array_api.dtypes import get_dtype_info
from pycv._lib.filters_support.kernel_utils import valid_offset, cast_kernel_dilation

__all__ = [
    'ctype_border_mode',
    'ctype_convex_hull_mode',
    'ctype_interpolation_order',
    'as_sequence',
    'fix_kernel_shape',
    'axis_transpose_to_last',
    'get_output',
    'get_kernel',
    'valid_kernel_shape_with_ref',
    'valid_same_shape',
    'valid_axis',
    'invert_values'
]


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


def ctype_convex_hull_mode(
        mode: str
) -> int:
    if mode == 'graham':
        return 1
    elif mode in ['jarvis', 'gift wrapping']:
        return 2
    else:
        raise RuntimeError('convex hull mode not supported')


def ctype_interpolation_order(
        order: int | str
) -> int:
    if isinstance(order, str):
        if order in ['nearest neighbor', 'nn']:
            return 0
        elif order == 'linear':
            return 1
        elif order == 'quadratic':
            return 2
        elif order == 'cubic':
            return 3
        else:
            raise RuntimeError('invalid interpolation order')
    elif not (0 <= order <= 3):
        raise ValueError('Order need to be in range of 0 - 3')
    return order


def ctype_hough_mode(
        mode: str
) -> int:
    if mode == 'line':
        return 1
    elif mode == 'circle':
        return 2
    elif mode in ['probabilistic_line', 'pp_line']:
        return 3
    else:
        raise RuntimeError('invalid hough mode')


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


def fix_kernel_shape(
        shape: tuple | int,
        axis: tuple,
        nd: int
) -> tuple:
    if nd < len(axis):
        raise RuntimeError('n dimensions is smaller then axis size')

    if isinstance(shape, numbers.Number):
        axis = valid_axis(nd, axis, 1)
        kernel_shape = tuple(shape if ax in axis else 1 for ax in range(nd))
    else:
        axis = valid_axis(nd, axis, len(shape))
        iter_shape = iter(shape)
        kernel_shape = tuple(next(iter_shape) if ax in axis else 1 for ax in range(nd))

    return kernel_shape


def axis_transpose_to_last(
        nd: int,
        axis: tuple | None,
        default_nd: int
) -> tuple[bool, tuple, tuple]:
    need_transpose = False
    transpose_forward = tuple()
    transpose_back = tuple()

    if axis is not None:
        if len(axis) != default_nd:
            raise ValueError(f'axis need to be None or tuple of ints with size of {default_nd}')
        axis = valid_axis(nd, axis, default_nd)

        for i, ax in enumerate(axis):
            if ax != nd - default_nd + i:
                need_transpose = True
                break

    if need_transpose:
        prev_ax = 0

        for ax in range(nd):
            if ax not in axis:
                transpose_forward += (ax,)
                transpose_back += (prev_ax,)
                prev_ax += 1
            else:
                for i, _ax in enumerate(axis):
                    if _ax == ax:
                        transpose_back += (nd - default_nd + i,)
                        break
        transpose_forward += axis

    return need_transpose, transpose_forward, transpose_back


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
        kernel_shape = fix_kernel_shape(kernel.size, (axis,), array_nd)
        kernel = np.reshape(kernel, kernel_shape)
        offset_scalar, offset = offset[0], tuple()
        for i in range(array_nd):
            offset += (0,) if i != axis else (offset_scalar,)
    elif array_nd != filter_dim:
        kernel = atleast_nd(kernel, array_nd, raise_err=False, expand_pos=0)
        offset = (0,) * (array_nd - filter_dim) + offset

    if not kernel.flags.contiguous:
        kernel = kernel.copy(order='C')

    return kernel, offset


########################################################################################################################

def valid_kernel_shape_with_ref(
        kernel_shape: tuple,
        image_shape: tuple
) -> None:
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


def valid_same_shape(
        *arrays: np.ndarray
) -> bool:
    if len(arrays) == 0: return True
    shape = arrays[0].shape
    for arr in arrays[1:]:
        if shape != arr.shape:
            return False
    return True


def valid_axis(
        nd: int,
        axis: Iterable | int | None,
        default_nd: int
) -> tuple:
    if axis is None:
        out = tuple()
        for i in range(min(default_nd, nd), 0, -1):
            out += (nd - i,)
    elif isinstance(axis, numbers.Number):
        out = (axis % nd if axis < 0 else axis,)
    elif isinstance(axis, Iterable):
        for ax in axis:
            if ax < -nd or ax > nd - 1:
                raise ValueError(f'axis {ax} is out of range for array with {nd} dimensions')
        out = tuple(ax % nd if ax < 0 else ax for ax in axis)
    else:
        raise ValueError('axis must be an int, iterable of ints, or None')
    if len(tuple(set(out))) != len(out):
        raise ValueError("axis must be unique")
    return out


########################################################################################################################

def invert_values(
        inputs: np.ndarray
) -> np.ndarray:
    dt = get_dtype_info(inputs.dtype)

    if dt.kind == 'b':
        out = ~inputs
    elif dt.kind == 'u':
        out = np.subtract(dt.max_val, inputs, dtype=inputs.dtype)
    else:
        out = -inputs

    return out

########################################################################################################################
