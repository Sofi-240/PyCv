import numpy as np
from pycv._lib._src_py import pycv_morphology, pycv_convexhull

__all__ = [
    'region_fill',
    'im_label',
    'convex_hull'
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
        return pycv_morphology.binary_region_fill(image, seed_point, strel, offset, output, inplace)

    seed_value = image[seed_point]

    inputs = np.where((image >= seed_value - value_tol) & (image <= seed_value + value_tol), False, True)

    pycv_morphology.binary_region_fill(inputs, seed_point, strel, offset, None, True)

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
    return pycv_morphology.labeling(image, connectivity, rng_mapping_method, mod_value)


########################################################################################################################


def convex_hull(
        image: np.ndarray,
        mask: np.ndarray | None = None,
        objects: bool = False,
        labels: np.ndarray | None = None,
        convex_image: bool = True
) -> tuple[np.ndarray] | tuple[np.ndarray, np.ndarray]:
    if objects:
        if labels is None:
            _, labels = im_label(image)
        if labels.dtype.kind == 'f':
            raise TypeError('labels image cannot be type of float')
        uni = np.unique(labels[labels != 0])
        if mask is None:
            mask = np.ones_like(labels, bool)
        elif mask.shape != labels.shape:
            raise ValueError('mask shape need to be same as labels shape')
        inputs = np.stack([np.asarray((labels == u) & mask, dtype=np.uint8) for u in uni], axis=0)
        mask = None
    else:
        inputs = image

    return pycv_convexhull.convex_hull_2d(inputs, mask, convex_image=convex_image)

########################################################################################################################
