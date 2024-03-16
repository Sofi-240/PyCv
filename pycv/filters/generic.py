import numpy as np
from .._lib.array_api.dtypes import cast, as_binary_array
from .._lib.array_api.regulator import np_compliance
from .._lib.array_api.array_pad import pad, get_padding_width
from .._lib._src_py.utils import valid_axis, fix_kernel_shape
from .._lib._src_py import pycv_filters
from .._lib.filters_support._windows import gaussian_kernel
from .._lib._src_py import pycv_morphology

__all__ = [
    'gaussian_filter',
    'mean_filter',
    'image_filter',
    'median_filter',
    'rank_filter',
    'local_min_filter',
    'local_max_filter'
]


########################################################################################################################

def _valid_footprint(
        ndim: int,
        kernel_shape: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
) -> np.ndarray:
    if footprint is not None:
        footprint = as_binary_array(footprint, 'footprint')
        kernel_shape = footprint.shape
        if footprint.ndim != ndim and axis is not None:
            axis = valid_axis(ndim, axis, footprint.ndim)
            if len(axis) != footprint.ndim:
                raise ValueError('footprint ndim dont match axis len')
            kernel_shape = fix_kernel_shape(kernel_shape, axis, ndim)
            footprint = np.reshape(footprint, kernel_shape)
    else:
        if kernel_shape is None:
            raise ValueError('one of kernel_shape or footprint must be given')
        elif np.isscalar(kernel_shape):
            kernel_shape = (kernel_shape, )
        axis = valid_axis(ndim, axis, len(kernel_shape))
        kernel_shape = fix_kernel_shape(kernel_shape, axis, ndim)
        footprint = np.ones(kernel_shape, bool)
    return footprint


########################################################################################################################

def gaussian_filter(
        image: np.ndarray,
        sigma: float | tuple,
        truncate: float = 3.,
        axis: tuple | None = None,
        preserve_dtype: bool = True,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0
) -> np.ndarray:
    """
    Apply Gaussian filter to the input image.

    Gaussian filter is a widely used filter in image processing for smoothing images by reducing noise and details.

    Parameters:
        image (numpy.ndarray): Input image to which the Gaussian filter will be applied.
        sigma (float or tuple): Standard deviation(s) of the Gaussian kernel. If a tuple is provided, it represents the standard deviation
            in each dimension. The higher the sigma, the more the image will be blurred.
        truncate (float, optional): Truncate the Gaussian filter at this many standard deviations. Defaults to 3.
        axis (tuple or None, optional): Specifies the axis or axes along which the Gaussian filter is applied. If None, the filter is applied to all axes. Defaults to None.
        preserve_dtype (bool, optional): If True, the dtype of the input image is preserved in the output. Defaults to True.
        padding_mode (str, optional): Specifies the padding mode for the convolution operation. Possible values are 'reflect', 'constant', 'edge', or 'symmetric'. Defaults to 'reflect'.
        constant_value (float or int or None, optional): Value to use for padding if padding_mode is set to 'constant'. If None, it defaults to 0. Defaults to 0.

    Returns:
        numpy.ndarray: Output image after applying the Gaussian filter.
    """
    image = np_compliance(image, 'image', _check_finite=True)
    dtype = image.dtype
    image = cast(image, np.float64)

    axis = valid_axis(image.ndim, axis, 2 if np.isscalar(sigma) else len(sigma))

    if np.isscalar(sigma):
        sigma = (sigma,) * len(axis)

    one_pass = len(set(sigma)) == 1
    if one_pass:
        kernel = gaussian_kernel(sigma[0], len(axis), truncate=truncate)

        kernel_shape = fix_kernel_shape(kernel.shape, axis, image.ndim)
        kernel = np.reshape(kernel, kernel_shape)

        output = pycv_filters.convolve(image, kernel, padding_mode=padding_mode, constant_value=constant_value)
    else:
        output = image.copy()
        for s, ax in zip(sigma, axis):
            kernel = gaussian_kernel(s, 1, truncate=truncate)
            output = pycv_filters.convolve(
                output, kernel, axis=ax, padding_mode=padding_mode, constant_value=constant_value
            )

    if preserve_dtype:
        output = cast(output, dtype)

    return output


########################################################################################################################

def mean_filter(
        image: np.ndarray,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        preserve_dtype: bool = True,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0
) -> np.ndarray:
    """
    Apply mean filter to the input image.

    Mean filter is a simple and commonly used filter for smoothing images by replacing each pixel value with the average value of its neighborhood.

    Parameters:
        image (numpy.ndarray): Input image to which the mean filter will be applied.
        kernel_size (int or tuple or None, optional): Size of the kernel for the mean filter. If None, the footprint parameter must be provided. Defaults to None.
        footprint (numpy.ndarray or None, optional): Custom footprint for the mean filter. If provided, kernel_size parameter will be ignored. Defaults to None.
        axis (int or tuple or None, optional): Specifies the axis or axes along which the mean filter is applied. Defaults to None.
        preserve_dtype (bool, optional): If True, the dtype of the input image is preserved in the output. Defaults to True.
        padding_mode (str, optional): Specifies the padding mode for the convolution operation. Possible values are 'reflect', 'constant', 'edge', or 'symmetric'. Defaults to 'reflect'.
        constant_value (float or int or None, optional): Value to use for padding if padding_mode is set to 'constant'. If None, it defaults to 0. Defaults to 0.

    Returns:
        numpy.ndarray: Output image after applying the mean filter.
    """
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)
    dtype = image.dtype
    image = cast(image, np.float64)

    footprint = _valid_footprint(image.ndim, kernel_size, footprint, axis)

    kernel = footprint.astype(np.float64)
    kernel /= np.sum(kernel)

    output = pycv_filters.convolve(image, kernel, padding_mode=padding_mode, constant_value=constant_value)

    if preserve_dtype:
        output = cast(output, dtype)

    return output


def image_filter(
        image: np.ndarray,
        kernel: np.ndarray,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0
) -> np.ndarray:
    """
    Apply custom image filter to the input image.

    Custom image filter is applied by convolving the input image with the provided kernel.

    Parameters:
        image (numpy.ndarray): Input image to which the custom image filter will be applied.
        kernel (numpy.ndarray): Custom kernel for the image filter.
        axis (int or tuple or None, optional): Specifies the axis or axes along which the filter is applied. Defaults to None.
        padding_mode (str, optional): Specifies the padding mode for the convolution operation. Possible values are 'reflect', 'constant', 'edge', or 'symmetric'. Defaults to 'reflect'.
        constant_value (float or int or None, optional): Value to use for padding if padding_mode is set to 'constant'. If None, it defaults to 0. Defaults to 0.

    Returns:
        numpy.ndarray: Output image after applying the custom image filter.
    """
    image = np_compliance(image, 'image', _check_finite=True)
    kernel = np_compliance(kernel, 'kernel', _check_finite=True)

    if image.ndim != kernel.ndim and axis is not None:
        axis = valid_axis(image.ndim, axis, kernel.ndim)
        if len(axis) != kernel.ndim:
            raise ValueError('kernel N dimensions dont match with axis length')

        for ax in range(image.ndim):
            if ax not in axis:
                kernel = np.expand_dims(kernel, ax)

    output = pycv_filters.convolve(image, kernel, padding_mode=padding_mode, constant_value=constant_value)

    return output


########################################################################################################################

def median_filter(
        image: np.ndarray,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    """
    Apply median filter to the input image.

    Median filter is a nonlinear filter used for noise reduction in images by replacing each pixel value with the median value of its neighborhood.

    Parameters:
        image (numpy.ndarray): Input image to which the median filter will be applied.
        kernel_size (int or tuple or None, optional): Size of the kernel for the median filter. If None, the footprint parameter must be provided. Defaults to None.
        footprint (numpy.ndarray or None, optional): Custom footprint for the median filter. If provided, kernel_size parameter will be ignored. Defaults to None.
        axis (int or tuple or None, optional): Specifies the axis or axes along which the median filter is applied. Defaults to None.
        padding_mode (str, optional): Specifies the padding mode for the convolution operation. Possible values are 'reflect', 'constant', 'edge', or 'symmetric'. Defaults to 'reflect'.
        constant_value (float or int or None, optional): Value to use for padding if padding_mode is set to 'constant'. If None, it defaults to 0. Defaults to 0.

    Returns:
        numpy.ndarray: Output image after applying the median filter.
    """
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)

    footprint = _valid_footprint(image.ndim, kernel_size, footprint, axis)

    rank = np.sum(footprint) // 2

    output = pycv_filters.rank_filter(image, footprint, rank, padding_mode=padding_mode, constant_value=constant_value)

    return output


def rank_filter(
        image: np.ndarray,
        rank: int,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    """
    Apply rank filter to the input image.

    Rank filter is a nonlinear filter used for various image processing tasks, including noise reduction, edge detection, and image enhancement.
    It replaces each pixel value with the value of the rank-th element in its neighborhood.

    Parameters:
        image (numpy.ndarray): Input image to which the rank filter will be applied.
        rank (int): Rank of the element to be selected from the neighborhood.
        kernel_size (int or tuple or None, optional): Size of the kernel for the rank filter. If None, the footprint parameter must be provided. Defaults to None.
        footprint (numpy.ndarray or None, optional): Custom footprint for the rank filter. If provided, kernel_size parameter will be ignored. Defaults to None.
        axis (int or tuple or None, optional): Specifies the axis or axes along which the rank filter is applied. Defaults to None.
        padding_mode (str, optional): Specifies the padding mode for the convolution operation. Possible values are 'reflect', 'constant', 'edge', or 'symmetric'. Defaults to 'reflect'.
        constant_value (float or int or None, optional): Value to use for padding if padding_mode is set to 'constant'. If None, it defaults to 0. Defaults to 0.

    Returns:
        numpy.ndarray: Output image after applying the rank filter.
    """
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)

    footprint = _valid_footprint(image.ndim, kernel_size, footprint, axis)

    if rank > np.sum(footprint):
        raise ValueError('invalid rank higher then the sum of footprint')

    output = pycv_filters.rank_filter(image, footprint, rank, padding_mode=padding_mode, constant_value=constant_value)

    return output


########################################################################################################################


def local_min_filter(
        image: np.ndarray,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    """
    Apply local minimum filter to the input image.

    Local minimum filter is a morphological filter used for finding local minima in images by replacing each pixel value with the minimum value within its neighborhood.

    Parameters:
        image (numpy.ndarray): Input image to which the local minimum filter will be applied.
        kernel_size (int or tuple or None, optional): Size of the kernel for the local minimum filter. If None, the footprint parameter must be provided. Defaults to None.
        footprint (numpy.ndarray or None, optional): Custom footprint for the local minimum filter. If provided, kernel_size parameter will be ignored. Defaults to None.
        axis (int or tuple or None, optional): Specifies the axis or axes along which the local minimum filter is applied. Defaults to None.
        padding_mode (str, optional): Specifies the padding mode for the convolution operation. Possible values are 'reflect', 'constant', 'edge', or 'symmetric'. Defaults to 'reflect'.
        constant_value (float or int or None, optional): Value to use for padding if padding_mode is set to 'constant'. If None, it defaults to 0. Defaults to 0.

    Returns:
        numpy.ndarray: Output image after applying the local minimum filter.
    """
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)
    footprint = _valid_footprint(image.ndim, kernel_size, footprint, axis)

    if padding_mode not in ['constant', 'valid']:
        image = pad(image, get_padding_width(footprint.shape), mode=padding_mode)
        padding_mode = 'valid'

    output = pycv_morphology.gray_ero_or_dil(0, image, footprint, offset=tuple(s // 2 for s in footprint.shape), border_val=constant_value)

    if padding_mode == 'valid':
        pw = get_padding_width(footprint.shape)
        output = output[tuple(slice(s[0], sh - s[1]) for (s, sh) in zip(pw, image.shape))]

    return output


def local_max_filter(
        image: np.ndarray,
        kernel_size: int | tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
) -> np.ndarray:
    """
    Apply local maximum filter to the input image.

    Local maximum filter is a morphological operation used for various image processing tasks, including noise reduction,
    edge detection, and image enhancement. It replaces each pixel value with the maximum value within its neighborhood.

    Parameters:
        image (numpy.ndarray): Input image to which the local maximum filter will be applied.
        kernel_size (int or tuple or None, optional): Size of the kernel for the local maximum filter. If None, the footprint parameter must be provided. Defaults to None.
        footprint (numpy.ndarray or None, optional): Custom footprint for the local maximum filter. If provided, kernel_size parameter will be ignored. Defaults to None.
        axis (int or tuple or None, optional): Specifies the axis or axes along which the local maximum filter is applied. Defaults to None.
        padding_mode (str, optional): Specifies the padding mode for the convolution operation. Possible values are 'reflect', 'constant', 'edge', or 'symmetric'. Defaults to 'reflect'.
        constant_value (float or int or None, optional): Value to use for padding if padding_mode is set to 'constant'. If None, it defaults to 0. Defaults to 0.

    Returns:
        numpy.ndarray: Output image after applying the local maximum filter.
    """
    if kernel_size is None and footprint is None:
        raise ValueError('one of the attribute kernel_size or footprint need to be given')

    image = np_compliance(image, 'image', _check_finite=True)
    footprint = _valid_footprint(image.ndim, kernel_size, footprint, axis)

    if padding_mode not in ['constant', 'valid']:
        image = pad(image, get_padding_width(footprint.shape), mode=padding_mode)
        padding_mode = 'valid'

    output = pycv_morphology.gray_ero_or_dil(1, image, footprint, offset=tuple(s // 2 for s in footprint.shape), border_val=constant_value)

    if padding_mode == 'valid':
        pw = get_padding_width(footprint.shape)
        output = output[tuple(slice(s[0], sh - s[1]) for (s, sh) in zip(pw, image.shape))]

    return output

########################################################################################################################
