import numpy as np
from ..array_api.regulator import np_compliance
from itertools import combinations_with_replacement
from .._src_py.pycv_filters import convolve
from .._src_py.utils import fix_kernel_shape
from pycv._lib.array_api.dtypes import cast
from ..filters_support._windows import EdgeKernels, gaussian_kernel

__all__ = [
    'derivatives'
]


########################################################################################################################

def derivatives(
        inputs: np.ndarray,
        sigma: float | tuple = 1.,
        padding_mode: str = 'constant',
        constant_value: float = 0.,
        mode_xy: bool = False
) -> np.ndarray:
    inputs = np_compliance(inputs, 'inputs', _check_finite=True, _check_atleast_nd=1)
    if inputs.dtype.kind == "f":
        inputs = inputs.astype(np.float64)
    else:
        inputs = cast(inputs, np.float64)

    ndim = inputs.ndim
    if ndim > 2 and mode_xy:
        mode_xy = False
    if np.isscalar(sigma):
        sigma = (sigma,) * ndim
    elif len(sigma) != ndim:
        raise ValueError('sigma need to be a float or tuple with size of ndim')

    kernel = EdgeKernels.SOBEL

    div = [
        convolve(
            inputs, kernel(ndim, i, normalize=False), padding_mode=padding_mode, constant_value=constant_value
        ) for i in range(ndim)
    ]

    if mode_xy:
        div = div[::-1]

    if len(set(sigma)) == 1:
        gauss_kernel = [gaussian_kernel(sigma[0], ndim)]
    else:
        gauss_kernel = []
        for i, s in enumerate(sigma):
            k = gaussian_kernel(s, 1)
            valid_shape = fix_kernel_shape(k.shape, axis=i, nd=ndim)
            gauss_kernel.append(k.reshape(valid_shape))

    tensor = np.zeros((ndim, ndim, *inputs.shape), np.float64)

    for i, j in combinations_with_replacement(range(ndim), 2):
        d = div[i] * div[j]
        if len(gauss_kernel) == 1:
            d = convolve(d, gauss_kernel[0], padding_mode=padding_mode, constant_value=constant_value)
        else:
            for k in gauss_kernel:
                d = convolve(d, k, padding_mode=padding_mode, constant_value=constant_value)
        tensor[i, j] = d
        if i != j:
            tensor[j, i] = d

    return tensor

########################################################################################################################
