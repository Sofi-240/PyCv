import numpy as np
from pycv._lib.core_support import morphology_py

__all__ = [
    'region_fill',
    'im_label'
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
        return morphology_py.binary_region_fill(image, seed_point, strel, offset, output, inplace)

    seed_value = image[seed_point]

    inputs = np.where((image >= seed_value - value_tol) & (image <= seed_value + value_tol), False, True)

    morphology_py.binary_region_fill(inputs, seed_point, strel, offset, None, True)

    if inplace:
        output = image
    elif not output:
        output = np.zeros_like(image)

    output[inputs] = fill_value if fill_value is not None else seed_value

    return output


########################################################################################################################

def im_label(
        image: np.ndarray,
        connectivity: int = 1,
        rng_mapping_method: str = 'sqr',
        mod_value: int = 16
) -> tuple[int, np.ndarray]:
    return morphology_py.labeling(image, connectivity, rng_mapping_method, mod_value)