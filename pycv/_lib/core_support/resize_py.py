import numpy as np
from pycv._lib.core_support.utils import get_output, axis_transpose_to_last
from pycv._lib.core import ops

__all__ = [
    'resize_bilinear'
]


def ctype_resize_mode(
        mode: str
) -> int:
    if mode == 'bilinear':
        return 1
    elif mode in ['nearest_neighbour', 'nn']:
        return 2
    else:
        raise RuntimeError('resize mode not supported')


########################################################################################################################

def resize_2d(
        inputs: np.ndarray,
        height: int,
        width: int,
        mode: str,
        axis: tuple | None = None
) -> np.ndarray:
    if inputs.ndim < 2:
        raise ValueError('Inputs dimensions for Bilinear resize need to be at least 2D')

    if height == 0 or width == 0:
        raise ValueError('height or width cannot be zero')

    nd = inputs.ndim

    if any(s == 0 for s in inputs.shape):
        raise ValueError('Inputs cannot have zero shape')

    need_transpose, transpose_forward, transpose_back = axis_transpose_to_last(nd, axis, default_nd=2)

    if need_transpose:
        inputs = inputs.transpose(transpose_forward)

    out_shape = inputs.shape[:-2] + (height, width)
    output, _ = get_output(None, inputs, out_shape)

    mode = ctype_resize_mode(mode)
    ops.resize_image(inputs, output, mode)

    if need_transpose:
        output = output.transpose(transpose_back)

    return output

########################################################################################################################
