import numpy as np
from pycv._lib.filters_support.windows import SOBEL_EDGE, SOBEL_WEIGHTS, PREWITT_WEIGHTS, PREWITT_EDGE
from pycv.filters._utils import edge_filters

__all__ = [
    'sobel',
    'prewitt',
    'PUBLIC'
]

PUBLIC = [
    'sobel',
    'prewitt',
]


########################################################################################################################

def sobel(
        image: np.ndarray,
        axis: tuple | None = None,
        padding_mode: str = 'reflect',
        **pad_kw
) -> np.ndarray:
    return edge_filters(image, SOBEL_WEIGHTS, SOBEL_EDGE, axis, preserve_dtype=False, padding_mode=padding_mode, **pad_kw)


def prewitt(
        image: np.ndarray,
        axis: tuple | None = None,
        padding_mode: str = 'reflect',
        **pad_kw
) -> np.ndarray:
    return edge_filters(image, PREWITT_WEIGHTS, PREWITT_EDGE, axis, preserve_dtype=False, padding_mode=padding_mode, **pad_kw)

