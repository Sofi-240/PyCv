import numpy as np
from ..misc.histogram import histogram, Histogram
from .._members_struct import function_members
from ..array_api.dtypes import cast
from ..array_api.regulator import np_compliance
from ..filters_support._windows import gaussian_kernel, sigma_from_size
from .._src_py.pycv_filters import convolve, rank_filter
from .._src_py.utils import as_sequence, valid_axis, fix_kernel_shape


__all__ = [
    "otsu",
    "li_and_lee",
    "kapur",
    "minimum_error",
    "minimum",
    "mean",
    "adaptive",
    "Thresholds"
]


########################################################################################################################

def _N1N2(
        hist: Histogram
) -> tuple[np.ndarray, np.ndarray]:
    n1 = np.cumsum(hist.counts[0])
    n2 = np.cumsum(hist.counts[0][::-1])[::-1]
    return n1, n2


def _Mu1Mu2(
        hist: Histogram,
        n1: np.ndarray | None = None,
        n2: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if n1 is None or n2 is None:
        n1, n2 = _N1N2(hist)
    mu1 = np.cumsum(hist.counts[0] * hist.bins) / n1
    mu2 = np.cumsum((hist.counts[0] * hist.bins)[::-1])[::-1] / n2
    return mu1, mu2


def _P1P2(
        hist: Histogram
) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.cumsum(hist.normalize[0])
    p2 = np.cumsum(hist.normalize[0][::-1])[::-1]
    return p1, p2


def _S1S2(
        hist: Histogram,
        n1: np.ndarray | None = None,
        n2: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n1 is None or n2 is None:
        n1, n2 = _N1N2(hist)
    mu1, mu2 = _Mu1Mu2(hist, n1, n2)

    s1 = np.sqrt(np.cumsum((hist.bins - mu1) ** 2) / n1)
    s2 = np.sqrt(np.cumsum(((hist.bins - mu2) ** 2)[::-1])[::-1] / n2)
    return s1, s2


def _non_zero_histogram(hist: Histogram):
    e1, e2 = hist.counts[0, 0], hist.counts[0, -1]
    if e1 == 0 or e2 == 0:
        c1 = np.argmax(hist.counts[0] > 0)
        c2 = hist.n_bins - np.argmax((hist.counts[0] > 0)[::-1])
        hist.__init__(hist.counts[:, c1:c2], hist.bins[c1:c2])


def _get_histogram(image: np.ndarray, nbin: int | None = None) -> Histogram:
    if not isinstance(image, np.ndarray):
        raise TypeError('image must be type of numpy.ndarray')
    image = np_compliance(image, 'Image', _check_finite=True)
    hist = histogram(image, bins=nbin)
    _non_zero_histogram(hist)
    return hist


########################################################################################################################


def otsu(image: np.ndarray, nbin: int | None = None) -> int | float:
    """
    Compute Otsu's threshold for thresholding an image.

    Parameters:
        image (np.ndarray): Input image, must be a 2D array (grayscale image).
        nbin (int | None, optional): Number of bins for histogram computation. If None, the number of bins is determined
            automatically. Defaults to None.

    Returns:
        int | float: Otsu's threshold value.

    Raises:
        ValueError: If the input image is not a 2D array (grayscale image).
        ValueError: If the number of bins is less than or equal to 1.

    References:
        1. Otsu, Nobuyuki. "A Threshold Selection Method from Gray-Level Histograms." IEEE Transactions on Systems,
           Man, and Cybernetics 9.1 (1979): 62-66.
        2. https://en.wikipedia.org/wiki/Otsu%27s_method
    """
    hist = _get_histogram(image, nbin)
    n1, n2 = _N1N2(hist)
    m1, m2 = _Mu1Mu2(hist, n1, n2)
    var = n1[:-1] * n2[1:] * (m1[:-1] - m2[1:]) ** 2
    th = hist.bins[np.argmax(var)]
    return th


def li_and_lee(image: np.ndarray, nbin: int | None = None) -> int | float:
    """
    Compute Li and Lee's threshold for thresholding an image.

    Parameters:
        image (np.ndarray): Input image, must be a 2D array (grayscale image).
        nbin (int | None, optional): Number of bins for histogram computation. If None, the number of bins is determined
            automatically. Defaults to None.

    Returns:
        int | float: Li and Lee's threshold value.

    Raises:
        ValueError: If the input image is not a 2D array (grayscale image).
        ValueError: If the number of bins is less than or equal to 1.

    References:
        1. Li, C. H., & Lee, C. K. (1993). Minimum cross entropy thresholding. Pattern Recognition, 26(4), 617-625.
        2. https://ieeexplore.ieee.org/document/251122
    """
    hist = _get_histogram(image, nbin)
    n1, n2 = _N1N2(hist)
    m1, m2 = _Mu1Mu2(hist, n1, n2)
    ni = (n1[:-1] * m1[:-1] * np.log(m1[:-1] + (m1[:-1] == 0))) + (n2[1:] * m2[1:] * np.log(m2[1:] + (m2[1:] == 0)))
    th = hist.bins[np.argmax(ni) + 1]
    return th


def kapur(image: np.ndarray, nbin: int | None = None) -> int | float:
    """
    Compute Kapur's entropy-based threshold for thresholding an image.

    Parameters:
        image (np.ndarray): Input image, must be a 2D array (grayscale image).
        nbin (int | None, optional): Number of bins for histogram computation. If None, the number of bins is determined
            automatically. Defaults to None.

    Returns:
        int | float: Kapur's threshold value.

    Raises:
        ValueError: If the input image is not a 2D array (grayscale image).
        ValueError: If the number of bins is less than or equal to 1.

    References:
        1. Kapur, J. N., Sahoo, P. K., & Wong, A. K. (1985). A new method for gray-level picture thresholding using
           the entropy of the histogram. Computer Vision, Graphics, and Image Processing, 29(3), 273-285.
    """
    hist = _get_histogram(image, nbin)
    p1, p2 = _P1P2(hist)
    h_norm = hist.normalize[0]

    ent = np.cumsum(h_norm * np.log(h_norm + (h_norm <= 0)))

    e1 = - (ent / p1) + np.log(p1)
    e2 = - ((ent[-1] - ent) / p2) + np.log(p2)

    fi = e1 + e2
    th = hist.bins[np.argmax(fi)]
    return th


def minimum_error(image: np.ndarray, nbin: int | None = None) -> int | float:
    """
    Compute the threshold using Minimum Error thresholding method.

    Parameters:
        image (np.ndarray): Input image, must be a 2D array (grayscale image).
        nbin (int | None, optional): Number of bins for histogram computation. If None, the number of bins is determined
            automatically. Defaults to None.

    Returns:
        int | float: Threshold value computed using the Minimum Error method.

    Raises:
        ValueError: If the input image is not a 2D array (grayscale image).
        ValueError: If the number of bins is less than or equal to 1.

    Note:
        This function internally computes the histogram of the input image and then applies the Minimum Error method to
        determine the optimal threshold.

    References:
        1. Kittler, J., & Illingworth, J. (1986). Minimum error thresholding. Pattern Recognition, 19(1), 41-47.
    """
    hist = _get_histogram(image, nbin)
    n1, n2 = _N1N2(hist)
    s1, s2 = _S1S2(hist, n1, n2)

    j1 = n1[1:] * np.log((s1[1:] + (s1[1:] == 0)) / (s2[:-1] + (s2[:-1] == 0)))
    j2 = n2[:-1] * np.log((s2[:-1] + (s2[:-1] == 0)) / (s1[1:] + (s1[1:] == 0)))

    j = 1 + 2 * (j1 + j2)
    th = hist.bins[np.argmin(j) + 1]
    return th


def minimum(image: np.ndarray, nbin: int | None = None, max_iterations: int = 10000) -> int | float:
    """
    Compute the threshold using the Minimum thresholding method.

    Parameters:
        image (np.ndarray): Input image, must be a 2D array (grayscale image).
        nbin (int | None, optional): Number of bins for histogram computation. If None, the number of bins is determined
            automatically. Defaults to None.
        max_iterations (int, optional): Maximum number of iterations to perform. Defaults to 10000.

    Returns:
        int | float: Threshold value computed using the Minimum thresholding method.

    Raises:
        ValueError: If the input image is not a 2D array (grayscale image).
        ValueError: If the number of bins is less than or equal to 1.
        RuntimeError: If 2 local maxima are not found in the histogram.
        RuntimeError: If the maximum number of iterations is reached without finding 2 local maxima.

    """
    hist = _get_histogram(image, nbin)
    bins = hist.bins
    n_bins = hist.n_bins

    h = np.ones((3,), ) / 3
    smooth = hist.counts[0]
    count = 0

    local_max = []

    while count < max_iterations:
        smooth = np.convolve(smooth, h, 'same')

        local_max.clear()
        look_for = 1

        for i in range(n_bins - 1):
            if look_for == 1 and smooth[i + 1] < smooth[i]:
                local_max.append(i)
                look_for = -1
            elif look_for == -1 and smooth[i + 1] > smooth[i]:
                look_for = 1

        if len(local_max) <= 2:
            break

        count += 1

    if len(local_max) != 2:
        raise RuntimeError(f'2 local maxima not fount in the histogram')

    if count == max_iterations:
        raise RuntimeError(f'reach maximum iterations, 2 local maxima not fount in the histogram')

    min_, max_ = local_max
    th = bins[np.argmin(smooth[min_:max_]) + min_]
    return th


########################################################################################################################

def mean(image: np.ndarray) -> float:
    """
    Compute the mean threshold value of pixel intensities in the image.

    Parameters:
        image (np.ndarray): Input image.

    Returns:
        float: Mean value of pixel intensities in the image.

    Raises:
        ValueError: If the input image is not a numpy array.

    """
    image = np_compliance(image, 'Image', _check_finite=True)
    return np.mean(image)


########################################################################################################################

ADAPTIVE_METHODS = {'gaussian', 'mean', 'median'}


def _adaptive_gaussian(
        inputs: np.ndarray,
        block_size: tuple,
        axis: tuple,
        sigma: tuple | float | None = None,
        padding_mode: str = 'reflect',
        constant_value: float = 0
) -> np.ndarray:
    if sigma is None:
        sigma = tuple(sigma_from_size(nn) for nn in block_size if nn != 1)
    else:
        try:
            sigma = as_sequence(sigma, len(axis))
        except RuntimeError:
            raise ValueError('sigma need to be a single float or a tuple of floats with size equal to filter dim')

    ndim = len(axis)

    if len(set(sigma)) == 1 and all(b == max(block_size) or b == 1 for b in block_size):
        kernel = gaussian_kernel(sigma[0], ndim=ndim, radius=max(block_size) // 2)
        kernel = np.reshape(kernel, fix_kernel_shape(kernel.shape, axis, ndim))
        threshold = convolve(inputs, kernel, padding_mode=padding_mode, constant_value=constant_value)
    else:
        threshold = inputs.copy()
        for s, a in zip(sigma, axis):
            kernel = gaussian_kernel(s, ndim=1, radius=block_size[a] // 2)
            threshold = convolve(threshold, kernel, axis=a, padding_mode=padding_mode, constant_value=constant_value)

    return threshold


def _adaptive_mean(
        inputs: np.ndarray,
        block_size: tuple,
        padding_mode: str = 'reflect',
        constant_value: float = 0
) -> np.ndarray:
    kernel = np.ones(block_size, dtype=np.float64) / np.prod(block_size)
    threshold = convolve(inputs, kernel, padding_mode=padding_mode, constant_value=constant_value)
    return threshold


def _adaptive_median(
        inputs: np.ndarray,
        block_size: tuple,
        padding_mode: str = 'reflect',
        constant_value: float = 0
) -> np.ndarray:
    rank = np.prod(block_size) // 2
    footprint = np.ones(block_size, bool)
    threshold = rank_filter(inputs, footprint, rank, padding_mode=padding_mode, constant_value=constant_value)
    return threshold


def adaptive(
        image: np.ndarray,
        block_size: tuple | int,
        method: str = 'gaussian',
        method_params=None,
        offset_val: int | float = 0,
        padding_mode: str = 'reflect',
        constant_value: float = 0,
        axis: tuple | None = None
) -> np.ndarray:
    """
    Apply adaptive thresholding to the input image.

    Parameters:
        image (np.ndarray): Input image.
        block_size (tuple | int): Size of the local neighborhood for computing the threshold. If an integer is provided,
            the same size will be used along all dimensions.
        method (str, optional): Method used for adaptive thresholding. Supported methods are 'gaussian', 'mean', and 'median'.
            Defaults to 'gaussian'.
        method_params (dict, optional): Additional parameters for the chosen method sigma for 'gaussian'. Defaults to None.
        offset_val (int | float, optional): Offset value added to the computed threshold. Defaults to 0.
        padding_mode (str, optional): Padding mode for handling borders. Supported modes are 'reflect', 'constant', 'edge',
            'symmetric', and 'wrap'. Defaults to 'reflect'.
        constant_value (float, optional): Constant value used in 'constant' padding mode. Defaults to 0.
        axis (tuple | None, optional): Axes along which the local neighborhoods are computed. If None, the computation
            is performed along all dimensions. Defaults to None.

    Returns:
        np.ndarray: Thresholded image.

    Raises:
        ValueError: If 'valid' padding mode is used, which is not supported for adaptive thresholding.

    """
    if method not in ADAPTIVE_METHODS:
        raise ValueError(f'{method} is not in supported methods use {ADAPTIVE_METHODS}')
    if padding_mode == 'valid':
        raise ValueError('valid padding is not supported for adaptive threshold')

    if not isinstance(image, np.ndarray):
        raise TypeError('image must be type of numpy.ndarray')

    image = np_compliance(image, 'Image', _check_finite=True)

    dtype = image.dtype
    casted = True
    need_float = method != 'median'
    if need_float and dtype.kind != 'f':
        image = cast(image, np.float64)
    elif need_float and dtype.itemsize != 8:
        image = image.astype(np.float64)
    else:
        casted = False

    if axis is None and np.isscalar(block_size):
        block_size = (block_size,) * min(2, image.ndim)
    elif np.isscalar(block_size):
        block_size = (block_size,) * len(axis)

    axis = valid_axis(image.ndim, axis, len(block_size))
    block_size = fix_kernel_shape(block_size, axis, image.ndim)

    if method == 'gaussian':
        threshold = _adaptive_gaussian(image, block_size, axis, method_params, padding_mode, constant_value)
    elif method == 'mean':
        threshold = _adaptive_mean(image, block_size, padding_mode, constant_value)
    else:
        threshold = _adaptive_median(image, block_size, padding_mode, constant_value)

    if casted:
        threshold = cast(threshold, dtype)

    return threshold - offset_val


########################################################################################################################


class Thresholds(function_members):
    OTSU = otsu
    LI_AND_LEE = li_and_lee
    KAPUR = kapur
    MINIMUM_ERROR = minimum_error
    MINIMUM = minimum
    MEAN = mean
    ADAPTIVE = adaptive

########################################################################################################################
