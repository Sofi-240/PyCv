import numbers
import numpy as np
import typing
from pycv._lib.histogram import HIST
from pycv._lib.decorator import registrate_decorator
from pycv._lib.filters_support.windows import gaussian_kernel
from pycv._lib.filters_support.filters import c_convolve, c_rank_filter
from cvpy._lib._inspect import isfunction
from cvpy._lib.array_api.dtypes import cast

__all__ = [
    'PUBLIC',
    'otsu',
    'li_and_lee',
    'kapur',
    'minimum_error',
    'mean',
    'HIST_TYPE',
    'ARRAY_TYPE',
    'BLOCK_TYPE',
    'METHODS'
]

PUBLIC = [

]
########################################################################################################################

HIST_TYPE = 1
ARRAY_TYPE = 2
BLOCK_TYPE = 3

METHODS = {}


def registrate_method(
        method_type: int,
        *functions
) -> None:
    for func in functions:
        if not isfunction(func):
            raise TypeError('func need to be type of function')
        name = func.__name__
        METHODS[name] = (method_type, func)


########################################################################################################################

@registrate_decorator(kw_syntax=True)
def nonzero_histogram(
        func, *args, **kwargs,
):
    hist = args[0]
    if not isinstance(hist, HIST):
        raise ValueError(f'hist need to be type of HIST')
    hist_, bins = hist[:]
    e1, e2 = hist_[0], hist_[-1]

    if e1 == 0 or e2 == 0:
        cond = hist_ > 0
        c1 = np.argmax(cond)
        c2 = hist_.size - np.argmax(cond[::-1])
        hist._replace(**{'hist': hist_[c1:c2], 'bins': bins[c1:c2]})

    return func(hist, *args[1:], **kwargs)


########################################################################################################################
# Thresholds by Histogram
@nonzero_histogram
def otsu(
        hist: HIST,
) -> int | float:
    hist, bins = hist[:]
    n1 = np.cumsum(hist)
    n2 = np.cumsum(hist[::-1])[::-1]

    m1 = np.cumsum(hist * bins) / n1
    m2 = np.cumsum((hist * bins)[::-1])[::-1] / n2

    var = n1[:-1] * n2[1:] * (m1[:-1] - m2[1:]) ** 2
    th = bins[np.argmax(var)]
    return th


@nonzero_histogram
def li_and_lee(
        hist: HIST,
) -> int | float:
    hist, bins = hist[:]

    n1 = np.cumsum(hist)
    n2 = np.cumsum(hist[::-1])[::-1]

    m1 = np.cumsum(hist * bins) / n1
    m2 = np.cumsum((hist * bins)[::-1])[::-1] / n2

    ni = (n1[:-1] * m1[:-1] * np.log(m1[:-1] + (m1[:-1] == 0))) + (n2[1:] * m2[1:] * np.log(m2[1:] + (m2[1:] == 0)))

    th = bins[np.argmax(ni) + 1]
    return th


@nonzero_histogram
def kapur(
        hist: HIST,
) -> int | float:
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


@nonzero_histogram
def minimum_error(
        hist: HIST,
) -> int | float:
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


@nonzero_histogram
def minimum(
        hist: HIST,
        max_iterations: int = 10000
) -> int | float:
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


registrate_method(HIST_TYPE, otsu, li_and_lee, kapur, minimum_error, minimum)


########################################################################################################################
# Thresholds on Array

def mean(
        image: np.ndarray
) -> float:
    return np.mean(image)


registrate_method(ARRAY_TYPE, mean)


########################################################################################################################

# Thresholds on block

def adaptive(
        image: np.ndarray,
        kernel_size: tuple,
        method: str = 'gaussian',
        method_params: typing.Any = None,
        offset_val: int | float = 0,
        padding_mode: str = 'reflect',
        **pad_kw
) -> np.ndarray:
    supported_mode = {'mean', 'gaussian', 'median', 'function'}

    if padding_mode == 'valid':
        raise ValueError('valid padding is not supported for adaptive threshold')

    if len(kernel_size) != 2:
        raise ValueError('kernel_size need to be size of 2')

    if method == 'function':
        raise RuntimeError(f'{method} is currently unsupported')

    elif method == 'gaussian':
        if method_params is None:
            sigma = lambda n: 0.3 * (n / 2 - 1) + 0.8
            method_params = tuple(sigma(nn) for nn in kernel_size)
        elif isinstance(method_params, numbers.Number):
            method_params = (method_params, method_params)

        if len(method_params) != 2:
            raise ValueError('method_params: (sigma) need to be a single float or a tuple of floats with size of 2')

        threshold = cast(image, np.float64)

        ndim = image.ndim
        for s, r, ax in zip(method_params, kernel_size, (ndim - 2, ndim - 1)):
            kernel = gaussian_kernel(s, radius=r)
            threshold = c_convolve(threshold, kernel, axis=(ax,), padding_mode=padding_mode, **pad_kw)
        threshold = cast(threshold, image.dtype)

    elif method == 'mean':
        kernel = np.ones(kernel_size, np.float64) / np.prod(kernel_size)
        threshold = cast(image, np.float64)
        threshold = c_convolve(threshold, kernel, padding_mode=padding_mode, **pad_kw)
        threshold = cast(threshold, image.dtype)
    elif method == 'median':
        rank = np.prod(kernel_size) // 2
        footprint = np.ones(kernel_size, bool)
        threshold = c_rank_filter(image, footprint, rank, padding_mode=padding_mode, **pad_kw)
    else:
        raise ValueError(f'{method} is not in supported methods use {supported_mode}')

    return threshold - offset_val


registrate_method(BLOCK_TYPE, adaptive)
########################################################################################################################
