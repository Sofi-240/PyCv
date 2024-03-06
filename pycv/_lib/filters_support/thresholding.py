import numpy as np
from pycv._lib.misc.histogram import bin_count, histogram, HIST
from pycv._lib.decorator import registrate_decorator
from pycv._lib.array_api.dtypes import cast
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.filters_support.windows import gaussian_kernel, sigma_from_size
from pycv._lib._src_py.pycv_filters import convolve, rank_filter
from pycv._lib._src_py.utils import as_sequence, valid_axis, fix_kernel_shape

__all__ = [
    'otsu',
    'li_and_lee',
    'kapur',
    'minimum_error',
    'minimum',
    'mean',
    'adaptive',
    'Threshold'
]


########################################################################################################################

@registrate_decorator(kw_syntax=True)
def histogram_type(func, *args, **kwargs):
    if len(args) < 1 and 'image' not in kwargs:
        raise ValueError('missing image input')
    elif len(args) < 1:
        inputs = kwargs.pop('image')
    else:
        inputs = args[0]
        args = args[1:]

    inputs = np_compliance(inputs, 'Image', _check_finite=True)

    # create histogram
    if np.issubdtype(inputs.dtype, np.integer):
        hist = bin_count(inputs)
    else:
        hist = histogram(inputs)

    # non zero histogram

    hist_, bins = hist[:]
    e1, e2 = hist_[0], hist_[-1]

    if e1 == 0 or e2 == 0:
        cond = hist_ > 0
        c1 = np.argmax(cond)
        c2 = hist_.size - np.argmax(cond[::-1])
        hist._replace(**{'hist': hist_[c1:c2], 'bins': bins[c1:c2]})

    return func(hist, *args, **kwargs)


@registrate_decorator(kw_syntax=True)
def array_type(func, *args, **kwargs):
    if len(args) < 1 and 'image' not in kwargs:
        raise ValueError('missing image input')
    elif len(args) < 1:
        inputs = kwargs.pop('image')
    else:
        inputs = args[0]
        args = args[1:]

    inputs = np_compliance(inputs, 'Image', _check_finite=True)
    return func(inputs, *args, **kwargs)


########################################################################################################################


@histogram_type
def otsu(hist: HIST) -> int | float:
    hist, bins = hist[:]
    n1 = np.cumsum(hist)
    n2 = np.cumsum(hist[::-1])[::-1]

    m1 = np.cumsum(hist * bins) / n1
    m2 = np.cumsum((hist * bins)[::-1])[::-1] / n2

    var = n1[:-1] * n2[1:] * (m1[:-1] - m2[1:]) ** 2
    th = bins[np.argmax(var)]
    return th


@histogram_type
def li_and_lee(hist: HIST) -> int | float:
    hist, bins = hist[:]

    n1 = np.cumsum(hist)
    n2 = np.cumsum(hist[::-1])[::-1]

    m1 = np.cumsum(hist * bins) / n1
    m2 = np.cumsum((hist * bins)[::-1])[::-1] / n2

    ni = (n1[:-1] * m1[:-1] * np.log(m1[:-1] + (m1[:-1] == 0))) + (n2[1:] * m2[1:] * np.log(m2[1:] + (m2[1:] == 0)))

    th = bins[np.argmax(ni) + 1]
    return th


@histogram_type
def kapur(hist: HIST) -> int | float:
    hist, bins = hist[:]
    hist /= np.sum(hist)

    p1 = np.cumsum(hist)
    p2 = np.cumsum(hist[::-1])[::-1]

    ent = np.cumsum(hist * np.log(hist + (hist <= 0)))

    e1 = - (ent / p1) + np.log(p1)
    e2 = - ((ent[-1] - ent) / p2) + np.log(p2)

    fi = e1 + e2
    th = bins[np.argmax(fi)]
    return th


@histogram_type
def minimum_error(hist: HIST) -> int | float:
    hist, bins = hist[:]

    n1 = np.cumsum(hist)
    n2 = np.cumsum(hist[::-1])[::-1]

    m1 = np.cumsum(hist * bins) / n1
    m2 = np.cumsum((hist * bins)[::-1])[::-1] / n2

    s1 = np.sqrt(np.cumsum((bins - m1) ** 2) / n1)

    s2 = np.sqrt(np.cumsum(((bins - m2) ** 2)[::-1])[::-1] / n2)

    j1 = n1[1:] * np.log((s1[1:] + (s1[1:] == 0)) / (s2[:-1] + (s2[:-1] == 0)))
    j2 = n2[:-1] * np.log((s2[:-1] + (s2[:-1] == 0)) / (s1[1:] + (s1[1:] == 0)))

    j = 1 + 2 * (j1 + j2)

    th = bins[np.argmin(j) + 1]
    return th


@histogram_type
def minimum(hist: HIST, max_iterations: int = 10000) -> int | float:
    hist, bins = hist[:]

    n_bins = len(bins)

    h = np.ones((3,), ) / 3
    smooth = hist
    count = 0

    local_max = []

    while count < max_iterations:
        smooth = np.convolve(smooth, h, 'same')

        local_max.clear()
        look_for = 1

        for i in range(n_bins - 1):
            if look_for == 1 and smooth[i + 1] < smooth[i]:
                local_max.append(bins[i])
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
    th = np.argmin(smooth[min_:max_]) + min_
    return th


########################################################################################################################

@array_type
def mean(image: np.ndarray) -> float:
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


@array_type
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
    if method not in ADAPTIVE_METHODS:
        raise ValueError(f'{method} is not in supported methods use {ADAPTIVE_METHODS}')
    if padding_mode == 'valid':
        raise ValueError('valid padding is not supported for adaptive threshold')

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

class Threshold:
    OTSU = otsu
    LI_AND_LEE = li_and_lee
    KAPUR = kapur
    MINIMUM_ERROR = minimum_error
    MINIMUM = minimum
    MEAN = mean
    ADAPTIVE = adaptive

    @classmethod
    def get_method(cls, method: str):
        try:
            return getattr(cls, method.upper())
        except AttributeError:
            raise Exception(f'{method} method is not supported')

########################################################################################################################

