import numpy as np
import typing
from pycv._lib.histogram import bin_count, histogram, HIST
from pycv._lib.decorator import registrate_decorator
from pycv._lib.array_api.dtypes import cast
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.filters_support.windows import gaussian_kernel, sigma_from_size
from pycv._lib.core_support.filters_py import convolve, rank_filter
from pycv._lib.core_support.utils import as_sequence, valid_axis
from pycv._lib._inspect import isfunction

__all__ = [
    'otsu',
    'li_and_lee',
    'kapur',
    'minimum_error',
    'minimum',
    'mean',
    'adaptive'
]


########################################################################################################################

def get_histogram(
        inputs: np.ndarray,
) -> HIST:
    if np.issubdtype(inputs.dtype, np.integer):
        hist = bin_count(inputs)
    else:
        hist = histogram(inputs)
    return hist


########################################################################################################################
@registrate_decorator(kw_syntax=True)
def histogram_dispatcher(
        func, *args, **kwargs,
):
    inputs = np_compliance(args[0], 'Image', _check_finite=True)
    hist = get_histogram(inputs)
    return func(hist, *args[1:], **kwargs)


@registrate_decorator(kw_syntax=True)
def array_dispatcher(
        func, *args, **kwargs,
):
    inputs = np_compliance(args[0], 'Image', _check_finite=True)
    return func(inputs, *args[1:], **kwargs)


@registrate_decorator(kw_syntax=True)
def block_dispatcher(
        func, *args, **kwargs,
):
    inputs = np_compliance(args[0], 'Image', _check_finite=True)
    if inputs.ndim < 2:
        raise ValueError('image n dimensions need to be at least 2')

    block_size = args[1]

    if block_size is None:
        raise ValueError('missing block_size parameter')

    try:
        block_size = as_sequence(block_size, 2)
    except RuntimeError:
        raise ValueError('block_size need to be a single int or a tuple of ints with size of 2')

    if not all(s % 2 != 0 for s in block_size):
        raise ValueError('block dimensions length need to be odd')

    return func(inputs, block_size, *args[2:], **kwargs)


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
@histogram_dispatcher
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


@histogram_dispatcher
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


@histogram_dispatcher
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


@histogram_dispatcher
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


@histogram_dispatcher
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


########################################################################################################################
# Thresholds on Array
@array_dispatcher
def mean(
        image: np.ndarray
) -> float:
    return np.mean(image)


########################################################################################################################
# Thresholds on block

@block_dispatcher
def adaptive(
        image: np.ndarray,
        block_size: tuple | int,
        method: str = 'gaussian',
        method_params: typing.Any = None,
        offset_val: int | float = 0,
        padding_mode: str = 'reflect',
        constant_value: float | int | None = 0,
        axis: tuple | None = None
) -> np.ndarray:
    supported_mode = {'gaussian', 'mean', 'median'}
    if padding_mode == 'valid':
        raise ValueError('valid padding is not supported for adaptive threshold')

    axis = valid_axis(image.ndim, axis, 2)
    if len(axis) != 2:
        raise ValueError('axis need to be tuple of ints with size of 2 or None')

    if method == 'function':
        raise RuntimeError(f'{method} is currently unsupported')
    elif method == 'gaussian':
        if method_params is None:
            method_params = tuple(sigma_from_size(nn) for nn in block_size)
        else:
            try:
                method_params = as_sequence(method_params, 2)
            except RuntimeError:
                raise ValueError('method_params: (sigma) need to be a single float or a tuple of floats with size of 2')

        if block_size[0] == block_size[1]:
            kernel = gaussian_kernel(method_params[0], ndim=2, radius=block_size[0] // 2)
            kernel = kernel.reshape(tuple(1 if ax not in axis else block_size[0] for ax in range(image.ndim)))
            threshold = convolve(cast(image, np.float64), kernel, padding_mode=padding_mode,
                                 constant_value=constant_value)
        else:
            threshold = cast(image, np.float64)
            for s, r, ax in zip(method_params, block_size, axis):
                kernel = gaussian_kernel(s, radius=r // 2)
                threshold = convolve(threshold, kernel, axis=ax, padding_mode=padding_mode,
                                     constant_value=constant_value)
        threshold = cast(threshold, image.dtype)
    elif method == 'mean':
        threshold = cast(image, np.float64)
        iter_shape = iter(block_size)
        kernel_shape = tuple(next(iter_shape) if ax in axis else 1 for ax in range(image.ndim))
        kernel = np.ones(kernel_shape, np.float64) / np.prod(kernel_shape)
        threshold = convolve(threshold, kernel, padding_mode=padding_mode, constant_value=constant_value)
        threshold = cast(threshold, image.dtype)
    elif method == 'median':
        iter_shape = iter(block_size)
        kernel_shape = tuple(next(iter_shape) if ax in axis else 1 for ax in range(image.ndim))
        rank = np.prod(kernel_shape) // 2
        footprint = np.ones(kernel_shape, bool)
        threshold = rank_filter(image, footprint, rank, padding_mode=padding_mode, constant_value=constant_value)
    else:
        raise RuntimeError(f'{method} is not in supported methods use {supported_mode}')

    return threshold - offset_val

########################################################################################################################

