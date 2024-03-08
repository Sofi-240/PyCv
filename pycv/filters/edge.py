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
    """
        Applies the Sobel edge detection filter to the input image along the specified axis.

        Parameters:
            image (numpy.ndarray): Input image to which the Sobel filter will be applied.
            axis (tuple or None, optional): Specifies the axis or axes along which the Sobel filter is applied.
                                            If None, the filter is applied to all axes. Defaults to None.
            padding_mode (str, optional): Specifies the padding mode for the convolution operation.
                                          Possible values are 'symmetric', 'reflect', 'constant', or 'edge'.
                                          Defaults to 'symmetric'.
            constant_value (float or None, optional): Value to use for padding if padding_mode is set to 'constant'.
                                                      If None, it defaults to 0.0. Defaults to 0.0.

        Returns:
            numpy.ndarray: Magnitude image containing the result of applying the Sobel filter.
    """
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
    """
        Applies the Prewitt edge detection filter to the input image along the specified axis.

        Parameters:
            image (numpy.ndarray): Input image to which the Sobel filter will be applied.
            axis (tuple or None, optional): Specifies the axis or axes along which the Prewitt filter is applied.
                                            If None, the filter is applied to all axes. Defaults to None.
            padding_mode (str, optional): Specifies the padding mode for the convolution operation.
                                          Possible values are 'symmetric', 'reflect', 'constant', or 'edge'.
                                          Defaults to 'symmetric'.
            constant_value (float or None, optional): Value to use for padding if padding_mode is set to 'constant'.
                                                      If None, it defaults to 0.0. Defaults to 0.0.

        Returns:
            numpy.ndarray: Magnitude image containing the result of applying the Prewitt filter.
    """
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
    return canny_filter(
        image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold,
        as_percentile=as_percentile, mask=mask, padding_mode=padding_mode, constant_value=constant_value
    )
########################################################################################################################
