import numpy as np
import numbers
from pycv._lib.filters_support.utils import default_axis, fix_kernel_shape, valid_kernels, get_output
from pycv._lib.filters_support.kernel_utils import default_binary_strel
from pycv._lib.decorator import registrate_decorator
from pycv._lib.array_api.dtypes import as_binary_array
from pycv._lib.array_api.regulator import np_compliance, check_finite
from pycv._lib.core import ops

__all__ = [
    'default_strel',
    'RAVEL_ORDER',
    'FLIPPER',
    'PUBLIC'
]

PUBLIC = []

RAVEL_ORDER = 'C'
MAX_NDIM = 3
FLIPPER = (1, 0, 2)


########################################################################################################################

def default_strel(
        ndim: int,
        strel: np.ndarray | None,
        connectivity=1,
        hole=False,
        flip: bool = True,
        dtype_bool: bool = False,
        offset: tuple | None = None
) -> np.ndarray:

    if strel is None:
        strel = default_binary_strel(ndim, connectivity)
    else:
        if not isinstance(strel, np.ndarray):
            raise TypeError(f'Strel need to be type of numpy.ndarray')

        if offset is not None:
            if not isinstance(offset, tuple):
                raise TypeError(f'offset point need to be type of tuple got {type(offset)}')
            if len(offset) != strel.ndim:
                raise ValueError(
                    f'Number of dimensions in center and kernel do not match: {len(offset)} != {strel.ndim}')
            if not all(of < s for of, s in zip(offset, strel.shape)):
                raise ValueError(f'offset point is out of range for Strel with shape of {strel.shape}')
        else:
            if not all(s % 2 != 0 for s in strel.shape):
                raise ValueError('Structuring element dimensions length need to be odd or set offset point')
            offset = (s // 2 for s in strel.shape)

        if dtype_bool and strel.dtype != bool:
            raise ValueError(f'strel dtype need to be boolean')

        if strel.ndim != ndim:
            raise ValueError(
                f'Number of dimensions in strel and image does not match {strel.ndim} != {ndim}'
            )
        if flip:
            strel = np.flip(strel, FLIPPER[:strel.ndim]) if strel.ndim > 1 else np.flip(strel, 0)
        if hole:
            strel = strel.copy()
            strel[offset] = False

    return strel

########################################################################################################################

