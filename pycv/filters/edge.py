import numpy as np
from pycv._lib.filters_support.windows import SOBEL_EDGE, SOBEL_WEIGHTS, PREWITT_WEIGHTS, PREWITT_EDGE
from pycv.filters._filters import edge_filters
from pycv._lib.filters_support.canny_edge import canny_filter

__all__ = [
    'sobel',
    'prewitt',
    'canny',
]


########################################################################################################################

def sobel(
        image: np.ndarray,
        axis: tuple | None = None,
        padding_mode: str = 'symmetric',
        constant_value: float | None = 0.0
) -> np.ndarray:
    return edge_filters(
        image, SOBEL_WEIGHTS, SOBEL_EDGE, axis=axis, preserve_dtype=False,
        padding_mode=padding_mode, constant_value=constant_value
    )


def prewitt(
        image: np.ndarray,
        axis: tuple | None = None,
        padding_mode: str = 'symmetric',
        constant_value: float | None = 0.0
) -> np.ndarray:
    return edge_filters(
        image, PREWITT_WEIGHTS, PREWITT_EDGE, axis=axis, preserve_dtype=False,
        padding_mode=padding_mode, constant_value=constant_value
    )


########################################################################################################################

def canny(
        image: np.ndarray,
        sigma: float | tuple = 1.0,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
        as_percentile: bool = False,
        mask: np.ndarray | None = None,
        padding_mode: str = 'constant',
        constant_value: float | None = 0.0
) -> np.ndarray:
    return canny_filter(
        image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold,
        as_percentile=as_percentile, mask=mask, padding_mode=padding_mode, constant_value=constant_value
    )
########################################################################################################################
