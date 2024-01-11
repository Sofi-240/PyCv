import numpy as np
import numbers
from pycv._lib.array_api.regulator import check_finite
from pycv._lib.array_api.shapes import atleast_nd
from pycv._lib.filters_support.kernel_utils import cast_kernel_dilation, valid_offset

FLIPPER = (1, 0, 2)

__all__ = [
    'default_axis',
    'fix_kernel_shape',
    'valid_kernels',
    'get_output',
    'PUBLIC'
]
PUBLIC = []

MAX_NDIM = 3
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
        filter_dim_bound: int = MAX_NDIM
) -> tuple[np.ndarray, tuple, tuple]:
    if not check_finite(kernel):
        raise ValueError('Kernel must not contain infs or NaNs')
    filter_dim = kernel.ndim
    if filter_dim > filter_dim_bound:
        raise ValueError(f'Operation for {filter_dim_bound + 1}D or above is not supported, got rank of {filter_dim}')
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
    share_memory = numpy.may_share_memory(inputs, output)
    return output, share_memory

########################################################################################################################