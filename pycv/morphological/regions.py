import numpy as np
from pycv._lib.filters_support.morphology import c_binary_region_fill

__all__ = [
    'region_fill',
    'binary_edge',
    'PUBLIC'
]

PUBLIC = [
    'region_fill',
    'binary_edge',
]


########################################################################################################################

def region_fill(
        image: np.ndarray,
        seed_point: tuple,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        output: np.ndarray | None = None,
        inplace: bool = False,
        value_tol: int | float = 0,
        fill_value: int | float | None = None
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Image need to be type of numpy.ndarray')

    if len(seed_point) != image.ndim:
        raise ValueError('Number of dimensions in seed_point and img do not match')

    if not all(0 <= sp < s for sp, s in zip(seed_point, image.shape)):
        raise ValueError('Seed point is out of range')

    if image.dtype == bool:
        return c_binary_region_fill(image, seed_point, strel, offset, output, inplace)

    seed_value = image[seed_point]

    inputs = np.where((image >= seed_value - value_tol) & (image <= seed_value + value_tol), False, True)

    c_binary_region_fill(inputs, seed_point, strel, offset, None, True)

    if inplace:
        output = image
    elif not output:
        output = np.zeros_like(image)
    output[inputs] = fill_value if fill_value is not None else seed_value
    return output


########################################################################################################################