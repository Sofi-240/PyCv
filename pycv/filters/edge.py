import numpy as np
from .._lib.array_api.dtypes import cast
from .._lib.array_api.regulator import np_compliance
from .._lib._decorator import wrapper_decorator
from .._lib._src_py.utils import valid_axis
from .._lib._src_py.pycv_filters import convolve
from .._lib.filters_support._windows import EdgeKernels
from .._lib._src_py import pycv_morphology
from .._lib.filters_support.kernel_utils import border_mask
from .._lib.filters_support.canny_edge import canny_filter

__all__ = [
    'EdgeKernels',
    'edge_filter',
    'sobel',
    'prewitt',
    'canny'
]


########################################################################################################################

def edge_filter(
        image: np.ndarray,
        kernel: EdgeKernels,
        preserve_dtype: bool = False,
        axis: int | tuple | None = None,
        padding_mode: str = 'symmetric',
        constant_value: float | None = 0.0,
) -> np.ndarray:
    image = np_compliance(image, 'image', _check_finite=True)
    dtype = image.dtype
    image = cast(image, np.float64)

    axis = valid_axis(image.ndim, axis, 2)

    if not isinstance(kernel, EdgeKernels):
        raise TypeError('kernel need to be type of EdgeKernels')

    kernel_ = kernel(image.ndim, axis[0])

    magnitude = len(axis) > 1

    output = convolve(image, kernel_, padding_mode=padding_mode, constant_value=constant_value)
    if magnitude:
        output *= output
        for ax in axis[1:]:
            kernel_ = kernel(image.ndim, ax)
            tmp = convolve(image, kernel_, padding_mode=padding_mode, constant_value=constant_value)
            output += (tmp * tmp)
        output = np.sqrt(output) / np.sqrt(len(axis))

    if padding_mode == 'constant':
        kernel_shape = tuple(len(kernel._edge) if a in axis else 1 for a in range(image.ndim))
        mask = border_mask(image.shape, kernel_shape)
        output = pycv_morphology.gray_erosion(image, np.ones(kernel_shape), mask=mask)

    if preserve_dtype:
        output = cast(output, dtype)

    return output


########################################################################################################################

@wrapper_decorator
def _edge_dispatcher(func=None, kernel: EdgeKernels = None, preserve_dtype: bool = False, *args, **kwargs):
    if not isinstance(kernel, EdgeKernels):
        raise TypeError('kernel need to be type of EdgeKernels')
    return edge_filter(args[0], kernel, preserve_dtype, *args[1:], **kwargs)


########################################################################################################################

@_edge_dispatcher(kernel=EdgeKernels.SOBEL)
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
                                        If None, the filter is applied to the last 2 axis.
                                        Defaults to None.
        padding_mode (str, optional): Specifies the padding mode for the convolution operation.
                                      Possible values are 'symmetric', 'reflect', 'constant', or 'edge'.
                                      Defaults to 'symmetric'.
        constant_value (float or None, optional): Value to use for padding if padding_mode is set to 'constant'.
                                                  If None, it defaults to 0.0. Defaults to 0.0.

    Returns:
        numpy.ndarray: Magnitude image containing the result of applying the Sobel filter.
    """
    pass


@_edge_dispatcher(kernel=EdgeKernels.PREWITT)
def prewitt(
        image: np.ndarray,
        axis: tuple | None = None,
        padding_mode: str = 'symmetric',
        constant_value: float | None = 0.0
) -> np.ndarray:
    """
    Applies the Prewitt edge detection filter to the input image along the specified axis.

    Parameters:
        image (numpy.ndarray): Input image to which the Prewitt filter will be applied.
        axis (tuple or None, optional): Specifies the axis or axes along which the Prewitt filter is applied.
                                        If None, the filter is applied to the last 2 axis.
                                        Defaults to None.
        padding_mode (str, optional): Specifies the padding mode for the convolution operation.
                                      Possible values are 'symmetric', 'reflect', 'constant', or 'edge'.
                                      Defaults to 'symmetric'.
        constant_value (float or None, optional): Value to use for padding if padding_mode is set to 'constant'.
                                                  If None, it defaults to 0.0. Defaults to 0.0.

    Returns:
        numpy.ndarray: Magnitude image containing the result of applying the Prewitt filter.
    """
    pass


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
