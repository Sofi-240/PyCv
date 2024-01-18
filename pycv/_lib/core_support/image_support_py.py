import numpy as np
from pycv._lib.core_support.utils import get_output, valid_same_shape
from pycv._lib.core import ops

__all__ = [
    'PUBLIC'
]
PUBLIC = []


########################################################################################################################

def canny_nonmaximum_suppression(
        magnitude: np.ndarray,
        grad_y: np.ndarray,
        grad_x: np.ndarray,
        low_threshold: float,
        mask: np.ndarray | None
) -> np.ndarray:
    valid_shape_tuple = (magnitude, grad_y, grad_x)
    valid_shape_tuple += (mask,) if mask is not None else tuple()
    if not valid_same_shape(*valid_shape_tuple):
        raise RuntimeError('all the ndarray inputs need to have the same shape')
    output, _ = get_output(None, magnitude)
    ops.canny_nonmaximum_suppression(magnitude, grad_y, grad_x, low_threshold, mask, output)
    return output

########################################################################################################################
