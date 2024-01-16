import numpy as np
import numbers
from pycv._lib.filters_support.utils import default_axis, fix_kernel_shape, valid_kernels, get_output, valid_kernel_shape_with_ref
from pycv._lib.filters_support.kernel_utils import default_binary_strel, color_mapping_range
from pycv._lib.decorator import registrate_decorator
from pycv._lib.array_api.dtypes import as_binary_array, get_dtype_limits, get_dtype_info
from pycv._lib.array_api.regulator import np_compliance, check_finite
from pycv._lib.core import ops

__all__ = [
    'default_strel',
    'c_binary_erosion',
    'RAVEL_ORDER',
    'MAX_NDIM',
    'FLIPPER',
    'PUBLIC'
]

PUBLIC = []

RAVEL_ORDER = 'C'
MAX_NDIM = 3
FLIPPER = (1, 0, 2)


########################################################################################################################

def default_strel(
        strel: np.ndarray | None,
        nd: int,
        flip: bool = True,
        connectivity: int = 1,
        hole: bool = False,
        offset: tuple | None = None
) -> tuple[np.ndarray, tuple]:
    if strel is None:
        strel = default_binary_strel(nd, connectivity, hole)
        flip = False
        hole = False
    strel, strel_shape, offset = valid_kernels(strel, nd, flip, 1, offset)
    if hole:
        strel = strel.copy()
        strel[offset] = False
    return strel, offset

########################################################################################################################

def c_binary_erosion(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        invert: bool | int = False,
) -> np.ndarray | None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Image need to be type of numpy.ndarray')
    nd = image.ndim

    if not check_finite(image):
        raise ValueError('image must not contain infs or NaNs')

    if image.dtype != bool:
        image = as_binary_array(image, 'Image')

    strel, offset = default_strel(strel, nd, offset=offset)

    if strel.dtype != bool:
        strel = as_binary_array(strel, 'strel')

    valid_kernel_shape_with_ref(strel.shape, image.shape)

    input_output = output is not None

    output, share_memory = get_output(output, image, image.shape)

    if share_memory:
        hold_output = output
        output, _ = get_output(hold_output.dtype, image, image.shape)

    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError(f'mask need to be type of numpy.ndarray')

        if mask.dtype != bool:
            raise ValueError(f'mask need to have boolean dtype')

        if not (mask.ndim == image.ndim and mask.shape == image.shape):
            raise ValueError(f'image and mask shape does not match {image.shape} != {mask.shape}')

    if np.all(image == 0):
        output[...] = 0
    else:
        ops.binary_erosion(image, strel, output, offset, iterations, mask, int(invert))

    if share_memory:
        hold_output[...] = output
        output = hold_output

    return None if input_output else output

########################################################################################################################

def c_binary_region_fill(
        image: np.ndarray,
        seed_point: tuple,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        output: np.ndarray | None = None,
        inplace: bool = False
) -> np.ndarray | None:
    nd = image.ndim

    if not check_finite(image):
        raise ValueError('image must not contain infs or NaNs')

    if image.dtype != bool:
        image = as_binary_array(image, 'Image')

    strel, offset = default_strel(strel, nd, offset=offset)
    valid_kernel_shape_with_ref(strel.shape, image.shape)

    if inplace:
        output = image
    else:
        output, _ = get_output(output, image, image.shape)

    if np.all(image == 0):
        output[...] = 0
    else:
        ops.binary_region_fill(output, strel, seed_point, offset)

    return output

########################################################################################################################

def c_gray_ero_or_dil(
        op: int,
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
) -> np.ndarray | None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Image need to be type of numpy.ndarray')
    nd = image.ndim

    if not check_finite(image):
        raise ValueError('image must not contain infs or NaNs')

    strel, offset = default_strel(strel, nd, offset=offset)

    if strel.dtype == bool:
        non_flat_strel = None
    else:
        non_flat_strel = strel
        strel = np.ones_like(strel, bool)

    valid_kernel_shape_with_ref(strel.shape, image.shape)

    input_output = output is not None

    output, share_memory = get_output(output, image, image.shape)

    if share_memory:
        hold_output = output
        output, _ = get_output(hold_output.dtype, image, image.shape)

    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError(f'mask need to be type of numpy.ndarray')

        if mask.dtype != bool:
            raise ValueError(f'mask need to have boolean dtype')

        if not (mask.ndim == image.ndim and mask.shape == image.shape):
            raise ValueError(f'image and mask shape does not match {image.shape} != {mask.shape}')

    cast_val = get_dtype_limits(output.dtype)[op]

    if np.all(image == 0):
        output[...] = np.min(image - (np.min(non_flat_strel) if non_flat_strel is not None else 0)) if op == 0 else \
            np.max(image + (np.max(non_flat_strel) if non_flat_strel is not None else 0))
    else:
        op = ops.erosion if op == 0 else ops.dilation
        op(image, strel, non_flat_strel, output, offset, mask, cast_val)

    if share_memory:
        hold_output[...] = output
        output = hold_output

    return None if input_output else output

########################################################################################################################


def c_labeling(
        image: np.ndarray,
        connectivity: int = 1,
        rng_mapping_method: str = 'sqr',
        mod_value: int = 16
) -> tuple[int, np.ndarray]:
    if not check_finite(image):
        raise ValueError('image must not contain infs or NaNs')
    if connectivity < 1 or connectivity > image.ndim:
        raise ValueError(
            f'Connectivity value must be in the range from 1 (no diagonal elements are neighbors) '
            f'to ndim (all elements are neighbors)'
        )

    dt = get_dtype_info(image.dtype)
    if dt.kind == 'b':
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

    output = np.zeros(image.shape, np.uint8)
    ops.labeling(image, connectivity, values_map, output)
    return np.max(output), output

########################################################################################################################

def c_skeletonize(
        image: np.ndarray
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Image need to be type of numpy.ndarray')
    nd = image.ndim

    if not check_finite(image):
        raise ValueError('image must not contain infs or NaNs')

    if image.dtype != bool:
        image = as_binary_array(image, 'Image')

    output, _ = get_output(None, image, image.shape)

    ops.skeletonize(image, output)
    return output