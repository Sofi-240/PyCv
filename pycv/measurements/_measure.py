import numpy as np
from .._lib._src_py import pycv_filters
from ..morphological import binary_edge, Strel
from .._lib.array_api.regulator import np_compliance

__all__ = [
    'perimeter',
    'moments'
]


########################################################################################################################

def perimeter(label_image: np.ndarray) -> float:
    """
    https://studylib.net/doc/5847818/design-and-fpga-implementation-of-a-perimeter-estimator
    """
    edge = binary_edge(
        label_image != 0,
        edge_mode='inner',
        strel=Strel.DEFAULT_STREL(label_image.ndim, connectivity=1)
    )
    kernel = np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]], np.float64)
    accumulate = pycv_filters.convolve(
        edge.astype(np.float64), kernel, padding_mode='constant', constant_value=0.
    )
    accumulate = accumulate.astype(np.int64)
    hist = np.bincount(accumulate.ravel(), minlength=50)

    weights = np.zeros(50, dtype=np.float64)
    weights[[5, 7, 15, 17, 25, 27]] = 1
    weights[[21, 33]] = np.sqrt(2)
    weights[[13, 23]] = (1 + np.sqrt(2)) / 2

    return hist @ weights


def moments(image: np.ndarray, center: np.ndarray | None = None, order: int = 3) -> np.ndarray:
    image = np_compliance(image, 'label_image', _check_finite=True).astype(np.float64)
    ndim = image.ndim

    if center is None:
        center = np.zeros((ndim, ) + (1, ) * ndim, dtype=np.float64)
    else:
        center = np.array(center, dtype=np.float64)
        center = center.reshape(center.shape + (1, ) * ndim)
        if center.ndim != ndim + 1:
            raise ValueError('center need to be with size equal to ndim')

    pq = np.indices(image.shape, dtype=np.float64) - center
    pq = pq[..., np.newaxis] ** np.arange(order + 1, dtype=np.float64)
    pq = pq.reshape(pq.shape + (1,) * (ndim - 1))

    output = image.reshape(image.shape + (1, ) * ndim)

    for i in range(ndim):
        axis = pq[i]
        axis = np.moveaxis(axis, ndim, ndim + i)
        output = output * axis

    output = np.sum(output, tuple(range(ndim)))
    return output

########################################################################################################################
