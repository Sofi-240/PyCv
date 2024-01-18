import numpy as np
from pycv._lib.core_support.utils import get_output, valid_same_shape
from pycv._lib.core import ops
from pycv._lib.filters_support.kernel_utils import default_binary_strel

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


def canny_hysteresis_edge_tracking(
        strong_edge: np.ndarray,
        week_edge: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not valid_same_shape(strong_edge, week_edge):
        raise RuntimeError('all the ndarray inputs need to have the same shape')

    if strong_edge.dtype != bool:
        strong_edge = strong_edge.astype(bool)
    if week_edge.dtype != bool:
        week_edge = week_edge.astype(bool)

    strel = default_binary_strel(2, connectivity=1, hole=True)

    ops.canny_hysteresis_edge_tracking(strong_edge, week_edge, strel)
    return strong_edge, week_edge

########################################################################################################################
