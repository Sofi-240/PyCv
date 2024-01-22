import numpy as np
import collections
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.filters_support.kernel_utils import color_mapping_range
from pycv._lib.array_api.dtypes import get_dtype_info
from pycv._lib.core_support.utils import get_output, valid_same_shape
from pycv._lib.core import ops

__all__ = [
    'canny_nonmaximum_suppression',
    'MAXTREE',
    'build_max_tree'
]


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

MAXTREE = collections.namedtuple('max_tree', 'traverser, parent')


def build_max_tree(
        image: np.ndarray,
        connectivity: int = 1,
        use_mapping: bool = False,
        rng_mapping_method: str = 'sqr',
        mod_value: int = 16
) -> MAXTREE:
    image = np.asarray(image, order='C')
    image = np_compliance(image, 'image', _check_finite=True)

    if connectivity < 1 or connectivity > image.ndim:
        raise ValueError(
            f'Connectivity value must be in the range from 1 (no diagonal elements are neighbors) '
            f'to ndim (all elements are neighbors)'
        )

    dt = get_dtype_info(image.dtype)

    if dt.kind == 'b' or not use_mapping:
        values_map = None
    else:
        if dt.kind == 'f':
            if np.min(image) < -1.0 or np.max(image) > 1.0:
                image = image.astype(np.int64)
        min_, max_ = np.min(image), np.max(image)
        if min_ < 0 or max_ > 255:
            image = ((image - min_) / (max_ - min_)) * 255
        image = image.astype(np.uint8)
        values_map = color_mapping_range(image, method=rng_mapping_method, mod_value=mod_value)

    traverser = np.zeros((image.size,), np.int64)
    parent = np.zeros(image.shape, np.int64)

    ops.build_max_tree(image, traverser, parent, connectivity, values_map)

    return MAXTREE(traverser, parent)

########################################################################################################################
