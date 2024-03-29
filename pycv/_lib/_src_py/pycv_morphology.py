import numpy as np
from .._src_py.utils import get_output, get_kernel, valid_kernel_shape_with_ref, invert_values
from pycv._lib._src import c_pycv
from ..filters_support.kernel_utils import default_binary_strel, color_mapping_range
from ..array_api.dtypes import as_binary_array, get_dtype_info
from ..array_api.regulator import np_compliance
from ..array_api.array_pad import pad

__all__ = [
    'default_strel',
    'binary_erosion',
    'binary_region_fill',
    'gray_erosion',
    'labeling',
    'skeletonize',
    'remove_small_objects',
    'binary_hit_or_miss',
]


########################################################################################################################

def default_strel(
        strel: np.ndarray | None,
        nd: int,
        connectivity: int = 1,
        hole: bool = False,
        offset: tuple | None = None
) -> tuple[np.ndarray, tuple]:
    if strel is None:
        strel = default_binary_strel(nd, connectivity, hole)
        hole = False
    strel, offset = get_kernel(strel, nd, offset=offset)
    if hole:
        strel = strel.copy()
        strel[offset] = False
    return strel, offset


########################################################################################################################

def binary_erosion(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        invert: bool | int = False,
        border_val: int = 0,
        extra_memory: bool = True
) -> np.ndarray | None:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True, _check_atleast_nd=1)
    nd = image.ndim

    strel, offset = default_strel(strel, nd, offset=offset)

    input_output = output is not None

    output, share_memory = get_output(output, image, image.shape)

    hold_output = None

    if share_memory:
        hold_output = output
        output, _ = get_output(hold_output.dtype, image, image.shape)

    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError(f'mask need to be type of numpy.ndarray')

        if mask.dtype != bool:
            raise ValueError(f'mask need to have boolean dtype')

        if mask.shape != image.shape:
            raise ValueError(f'image and mask shape does not match {image.shape} != {mask.shape}')

    if (np.all(image == 0) and border_val == 0) or iterations == 0:
        output[...] = 0
    elif iterations != 1 and (strel[offset] == 0 or not extra_memory):
        inp = image.copy()
        change = True
        while iterations != 0 and change:
            c_pycv.binary_erosion(inp, strel, output, offset, 1, mask, int(invert), border_val)
            change = np.any(inp != output)
            inp[...] = output
            iterations -= 1
    else:
        c_pycv.binary_erosion(image, strel, output, offset, iterations, mask, int(invert), border_val)

    if share_memory:
        hold_output[...] = output
        output = hold_output

    return None if input_output else output


########################################################################################################################

def binary_region_fill(
        image: np.ndarray,
        seed_point: tuple,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        output: np.ndarray | None = None,
        inplace: bool = False
) -> np.ndarray:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True, _check_atleast_nd=1)

    nd = image.ndim

    strel, offset = default_strel(strel, nd, offset=offset, hole=True)

    if inplace:
        output = image
    else:
        output, _ = get_output(output, image, image.shape)
        output[...] = image

    if np.all(image == 0):
        output[...] = 1
    else:
        c_pycv.binary_region_fill(output, seed_point, strel, offset)

    return output


########################################################################################################################

def gray_erosion(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: float = 0,
        invert: bool = False
) -> np.ndarray | None:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True, _check_atleast_nd=1)
    nd = image.ndim

    strel, offset = default_strel(strel, nd, offset=offset)

    non_flat_strel = strel.dtype != bool
    input_output = output is not None

    output, share_memory = get_output(output, image, image.shape)
    hold_output = None

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

    if np.all(image == 0) and border_val == 0:
        output[...] = np.min(image - (np.min(strel) if non_flat_strel else 0)) if not invert else \
            np.max(image + (np.max(strel) if non_flat_strel else 0))
    else:
        c_pycv.gray_erosion(image, strel, output, offset, mask, int(invert), border_val)

    if share_memory:
        hold_output[...] = output
        output = hold_output

    return None if input_output else output


########################################################################################################################


def labeling(
        image: np.ndarray,
        connectivity: int = 1,
        rng_mapping_method: str = 'sqr',
        mod_value: int = 16,
) -> tuple[int, np.ndarray]:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True, _check_atleast_nd=1)

    if connectivity < 1 or connectivity > image.ndim:
        raise ValueError(
            f'Connectivity value must be in the range from 1 (no diagonal elements are neighbors) '
            f'to ndim (all elements are neighbors)'
        )

    dt = get_dtype_info(image.dtype)
    if dt.kind == 'b':
        inputs = image
    else:
        if dt.kind == 'f':
            if np.min(image) < -1.0 or np.max(image) > 1.0:
                image = image.astype(np.int64)
        min_, max_ = np.min(image), np.max(image)
        if min_ < 0 or max_ > 255:
            image = ((image - min_) / (max_ - min_)) * 255
        image = image.astype(np.uint8)
        values_map = color_mapping_range(image, method=rng_mapping_method, mod_value=mod_value)
        inputs = values_map[image]

    output = np.zeros(inputs.shape, np.int64)

    c_pycv.labeling(inputs.astype(np.int64, copy=False), connectivity, output)

    return np.max(output), output


########################################################################################################################

def skeletonize(
        image: np.ndarray
) -> np.ndarray:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True, _check_atleast_nd=2)
    pw = ((1, 1), ) * image.ndim
    inputs = pad(image, pw, mode='constant', constant_values=0)
    output = c_pycv.skeletonize(inputs)
    output = output[tuple(slice(1, -1) for _ in range(inputs.ndim))]
    return output


########################################################################################################################

def remove_small_objects(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1,
        invert: int = 0
) -> np.ndarray:
    inv_img = image.copy()
    bool_cast = False

    if image.dtype != bool:
        inv_img = as_binary_array(inv_img, 'Image')

    if invert:
        inv_img = ~inv_img

    n_labels, labels = labeling(inv_img, connectivity)

    area = np.bincount(labels.ravel())

    area_bool = np.where(area > threshold, True, False)
    area_bool[0] = False

    labels_bool = area_bool[labels]

    if invert:
        labels_bool = ~labels_bool

    if not bool_cast:
        return labels_bool

    output, _ = get_output(None, image)
    output[labels_bool] = np.max(image)

    return output


########################################################################################################################

def binary_hit_or_miss(
        image: np.ndarray,
        strel1: np.ndarray | None = None,
        strel2: np.ndarray | None = None,
        offset1: tuple | None = None,
        offset2: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray | None:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True)
    ndim = image.ndim

    if image.dtype != bool:
        image = as_binary_array(image, 'Image')

    strel1, offset1 = default_strel(strel1, ndim, offset=offset1)

    if strel1.dtype != bool:
        strel1 = as_binary_array(strel1, 'strel1')

    if strel2 is None:
        strel2 = np.logical_not(strel1)
        offset2 = offset1

    strel2, offset2 = default_strel(strel2, ndim, offset=offset2)

    if strel2.dtype != bool:
        strel2 = as_binary_array(strel2, 'strel2')

    valid_kernel_shape_with_ref(strel1.shape, image.shape)
    valid_kernel_shape_with_ref(strel2.shape, image.shape)

    input_output = output is not None
    output, share_memory = get_output(output, image, image.shape)

    hold_output = None
    if share_memory:
        hold_output = output
        output, _ = get_output(hold_output.dtype, image, image.shape)

    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError(f'mask need to be type of numpy.ndarray')

        if mask.dtype != bool:
            raise ValueError(f'mask need to have boolean dtype')

        if mask.shape != image.shape:
            raise ValueError(f'image and mask shape does not match {image.shape} != {mask.shape}')

    if np.all(image == 0) and border_val == 0:
        output[...] = 0
    else:
        tmp, _ = get_output(output.dtype, image, image.shape)
        c_pycv.binary_erosion(image, strel1, tmp, offset1, mask, 0, border_val)
        c_pycv.binary_erosion(image, strel2, output, offset2, mask, 0, border_val)
        np.logical_not(output, output)
        np.logical_and(tmp, output, output)

    if share_memory:
        hold_output[...] = output
        output = hold_output

    return None if input_output else output

########################################################################################################################
