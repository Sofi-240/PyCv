import numpy as np
from pycv._lib.array_api.dtypes import cast, get_dtype_limits
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.core_support.filters_py import convolve
from pycv._lib.filters_support.windows import gaussian_kernel, SOBEL_EDGE, SOBEL_WEIGHTS, edge_kernel
from pycv._lib.core_support.image_support_py import canny_nonmaximum_suppression, canny_hysteresis_edge_tracking
from pycv._lib.filters_support.kernel_utils import default_binary_strel, border_mask
from pycv._lib.core_support import morphology_py

__all__ = [
    'canny_filter',
    'PUBLIC'
]

PUBLIC = []


########################################################################################################################

def _smooth_image(
        image: np.ndarray,
        sigma: float,
        mask: np.ndarray | None,
        padding_mode: str,
        constant_value: float
) -> tuple[np.ndarray, np.ndarray | None]:
    image = cast(image, np.float64)
    kernel = gaussian_kernel(sigma, ndim=2)

    if mask is not None:
        mask = np_compliance(mask, 'Mask').astype(bool)
        if mask.shape != image.shape:
            raise RuntimeError(f'image and mask shape does not match {image.shape} != {mask.shape}')
        image_mask = np.zeros_like(image)
        image_mask[mask] = image[mask]

        morphology_py.binary_erosion(mask, default_binary_strel(2, 2), output=mask)
    else:
        image_mask = image
        if padding_mode == 'constant':
            mask = ~border_mask(image.shape, (3, 3), (1, 1))

    blur_image = convolve(image_mask, kernel, padding_mode=padding_mode, constant_value=constant_value)

    if mask is not None:
        mask = convolve(mask.astype(image.dtype), kernel, padding_mode=padding_mode, constant_value=constant_value)
        mask += np.finfo(image.dtype).eps
        blur_image /= mask

    return blur_image, mask


########################################################################################################################

def canny_filter(
        image: np.ndarray,
        sigma: float | tuple = 1.0,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
        threshold_quantiles: bool = False,
        mask: np.ndarray | None = None,
        padding_mode: str = 'constant',
        constant_value: float | None = 0.0
) -> np.ndarray:
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True)

    if image.ndim != 2:
        raise ValueError('Canny supported just for 2D arrays')

    if padding_mode == 'valid':
        raise ValueError('valid padding mode not supported for canny filter')

    max_val = get_dtype_limits(image.dtype)[1]

    if low_threshold is None:
        low_threshold = 0.1
    elif threshold_quantiles:
        if not (0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        low_threshold /= max_val

    if high_threshold is None:
        high_threshold = 0.2
    elif threshold_quantiles:
        if not (0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        high_threshold /= max_val

    if high_threshold < low_threshold:
        raise ValueError("low_threshold should be lower then high_threshold")

    blur_image, mask = _smooth_image(image, sigma, mask, padding_mode, constant_value)

    dy_kernel = edge_kernel(SOBEL_WEIGHTS, SOBEL_EDGE, 2, 0) * 4
    dx_kernel = edge_kernel(SOBEL_WEIGHTS, SOBEL_EDGE, 2, 1) * 4

    gy = convolve(blur_image, dy_kernel, padding_mode='symmetric')
    gx = convolve(blur_image, dx_kernel, padding_mode='symmetric')

    magnitude = np.hypot(gy, gx)

    if threshold_quantiles:
        low_threshold, high_threshold = np.percentile(magnitude, [100.0 * low_threshold, 100.0 * high_threshold])

    edges = canny_nonmaximum_suppression(magnitude, gy, gx, low_threshold, mask)

    strong_edge = edges >= high_threshold
    week_edge = (edges > 0) & ~strong_edge

    strong_edge, week_edge = canny_hysteresis_edge_tracking(strong_edge, week_edge)

    # edges_mask = edges > 0
    # n_labels, labels = morphology_py.labeling(edges_mask, connectivity=2)
    #
    # if n_labels == 1:
    #     return edges_mask
    #
    # labels_of_strong_edge = np.unique(labels[edges_mask & (edges >= high_threshold)])
    #
    # labels_bool = np.zeros((n_labels + 1,), bool)
    # labels_bool[labels_of_strong_edge] = True
    # strong_edge = labels_bool[labels]

    return strong_edge
