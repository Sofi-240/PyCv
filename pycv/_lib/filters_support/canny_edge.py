import numpy as np
from ..array_api.dtypes import cast, get_dtype_limits
from ..array_api.regulator import np_compliance
from .._src_py.pycv_filters import convolve
from ._windows import gaussian_kernel, EdgeKernels
from .._src_py.pycv_minsc import canny_nonmaximum_suppression
from .kernel_utils import default_binary_strel, border_mask
from .._src_py import pycv_morphology


__all__ = [
    'canny_filter'
]


########################################################################################################################

def _valid_threshold(
        image_dtype: np.dtype,
        low_threshold: float | None,
        high_threshold: float | None,
        as_percentile: bool
) -> tuple[float, float]:
    div = get_dtype_limits(image_dtype)[1]

    def routine(th, default):
        if th is None:
            th = default
        elif as_percentile:
            if not (0.0 <= th <= 1.0):
                raise ValueError("If as_quantiles is True thresholds must be between 0 and 1.")
        else:
            th /= div
        return th

    low_threshold = routine(low_threshold, 0.1)
    high_threshold = routine(high_threshold, 0.2)

    if high_threshold < low_threshold:
        raise ValueError("low_threshold should be lower then high_threshold")

    return low_threshold, high_threshold


def _smooth_image(
        image: np.ndarray,
        sigma: float,
        mask: np.ndarray | None,
        padding_mode: str,
        constant_value: float
) -> tuple[np.ndarray, np.ndarray | None]:
    image = cast(image, np.float64)

    kernel = gaussian_kernel(sigma, ndim=2, radius=2)

    if mask is not None:
        mask = np_compliance(mask, 'Mask').astype(bool)
        if mask.shape != image.shape:
            raise RuntimeError(f'image and mask shape does not match {image.shape} != {mask.shape}')
        image_mask = np.zeros_like(image)
        image_mask[mask] = image[mask]

        pycv_morphology.binary_erosion(mask, default_binary_strel(2, 2), output=mask)
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
        as_percentile: bool = False,
        mask: np.ndarray | None = None,
        padding_mode: str = 'constant',
        constant_value: float | None = 0.0
) -> np.ndarray:
    """
        Applies the Canny edge detection algorithm to the input image.

        Parameters:
            image (numpy.ndarray): Input image to which the Canny edge detection algorithm will be applied.
            sigma (float or tuple, optional): Standard deviation of the Gaussian filter used for image smoothing.
                                              If a tuple is provided, it represents the standard deviation
                                              in the x and y directions respectively. Defaults to 1.0.
            low_threshold (float or None, optional): Lower threshold for edge detection.
                                                     If None, it is automatically calculated based on the image intensity distribution.
                                                     Defaults to None.
            high_threshold (float or None, optional): Higher threshold for edge detection.
                                                      If None, it is automatically calculated based on the low_threshold value.
                                                      Defaults to None.
            as_percentile (bool, optional): If True, low_threshold and high_threshold are interpreted as percentiles of the image intensity distribution.
                                            Defaults to False.
            mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image.
                                                    If provided, only the edges within the mask will be detected.
                                                    Defaults to None.
            padding_mode (str, optional): Specifies the padding mode for the convolution operations.
                                          Possible values are 'constant', 'symmetric', 'reflect', or 'edge'.
                                          Defaults to 'constant'.
            constant_value (float or None, optional): Value to use for padding if padding_mode is set to 'constant'.
                                                      If None, it defaults to 0.0. Defaults to 0.0.

        Returns:
            numpy.ndarray: Output image containing the edges detected by the Canny algorithm.
    """
    image = np.asarray(image)
    image = np_compliance(image, 'image', _check_finite=True)

    if image.ndim != 2:
        raise ValueError('Canny supported just for 2D arrays')

    if padding_mode == 'valid':
        raise ValueError('valid padding mode not supported for canny filter')

    low_threshold, high_threshold = _valid_threshold(image.dtype, low_threshold, high_threshold, as_percentile)

    blur_image, mask = _smooth_image(image, sigma, mask, padding_mode, constant_value)

    dy_kernel = EdgeKernels.SOBEL.get_kernel(2, 0, normalize=False)
    dx_kernel = EdgeKernels.SOBEL.get_kernel(2, 1, normalize=False)

    gy = convolve(blur_image, dy_kernel, padding_mode='symmetric')
    gx = convolve(blur_image, dx_kernel, padding_mode='symmetric')
    magnitude = np.hypot(gy, gx)

    if as_percentile:
        low_threshold, high_threshold = np.percentile(magnitude, [100.0 * low_threshold, 100.0 * high_threshold])

    edges = canny_nonmaximum_suppression(magnitude, gy, gx, low_threshold, mask)

    edges_mask = edges > 0
    n_labels, labels = pycv_morphology.labeling(edges_mask, connectivity=2)

    if n_labels == 1:
        return edges_mask, None

    labels_of_strong_edge = np.unique(labels[edges_mask & (edges >= high_threshold)])

    labels_bool = np.zeros((n_labels + 1,), bool)
    labels_bool[labels_of_strong_edge] = True
    strong_edge = labels_bool[labels]

    return strong_edge
