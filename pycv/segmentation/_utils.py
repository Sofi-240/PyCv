import numpy as np
import numbers
from pycv._lib.array_api.regulator import np_compliance, check_finite
from pycv._lib.decorator import registrate_decorator
from pycv._lib.histogram import bin_count, histogram, HIST
from pycv._lib.filters_support.segmentation import METHODS, HIST_TYPE, ARRAY_TYPE, BLOCK_TYPE

__all__ = [
    'threshold_dispatcher',
    'get_histogram',
    'im_binarize',
    'im_threshold',
    'PUBLIC'
]

PUBLIC = [
    'im_binarize',
    'im_threshold',
]


########################################################################################################################

def get_histogram(
        image: np.ndarray,
) -> HIST:
    if np.issubdtype(image.dtype, np.integer):
        hist = bin_count(image)
    else:
        hist = histogram(image)
    return hist


def im_binarize(
        image: np.ndarray,
        threshold: int | float | np.ndarray,
) -> np.ndarray:
    return np.where(image > threshold, True, False)


########################################################################################################################

_PASS = '-'


@registrate_decorator(kw_syntax=True)
def threshold_dispatcher(
        func,
        method_name: str,
        return_image: bool = False,
        *args, **kwargs
):
    image = np_compliance(args[0])
    check_finite(image, raise_err=True)

    if method_name == _PASS:
        method_name = args[1]
        args = tuple(arg for i, arg in enumerate(args) if i != 1)

    method_type, call = METHODS.get(method_name, (None, None))

    if call is None:
        raise ValueError(f'unsupported method {method_name}')

    def out_func(t):
        if not return_image:
            return t
        return im_binarize(image, t), t

    if method_type == HIST_TYPE:
        threshold = call(get_histogram(image), *args[1:], **kwargs)
        return out_func(threshold)

    if method_type == ARRAY_TYPE:
        threshold = call(image, *args[1:], **kwargs)
        return out_func(threshold)

    if not method_type == BLOCK_TYPE:
        raise RuntimeError('method type is not supported')

    # BLOCK TYPE
    kernel_size = args[1] if len(args) > 1 else kwargs.get('kernel_size', None)

    if kernel_size is None:
        raise ValueError('missing kernel_size parameter')

    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, tuple) and len(kernel_size) != 2:
        raise ValueError(f'kernel_size need to be size of 2')
    elif not isinstance(kernel_size, tuple):
        raise TypeError('kernel_size need to be int or tuple of ints')

    if not all(s % 2 != 0 for s in kernel_size):
        raise ValueError('Kernel dimensions length need to be odd')

    if len(args) > 1:
        threshold = call(image, kernel_size, *args[2:], **kwargs)
    else:
        kwargs['kernel_size'] = kernel_size
        threshold = call(image, **kwargs)

    return out_func(threshold)


@threshold_dispatcher(method_name='-', return_image=True)
def im_threshold(
        image: np.ndarray,
        method: str,
        return_threshold: bool = True,
        *args, **kwargs
) -> np.ndarray | tuple[np.ndarray, int | float | np.ndarray]:
    pass

########################################################################################################################
