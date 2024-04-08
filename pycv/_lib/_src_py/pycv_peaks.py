import numpy as np
from ..array_api.regulator import np_compliance
from ..array_api import iterators
from .utils import get_output, ctype_border_mode, axis_transpose_to_last, invert_values, valid_axis
from pycv._lib._src import c_pycv


__all__ = [
    'peak_nonmaximum_suppression'
]


########################################################################################################################

def peak_nonmaximum_suppression(
        inputs: np.ndarray,
        min_distance: tuple | int,
        threshold: float | None = None,
        axis: tuple | None = None,
        padding_mode: str = 'constant',
        constant_value: float = 0,
        invert: bool = False,
) -> np.ndarray:

    inputs = np_compliance(inputs, 'Inputs', _check_finite=True, _check_atleast_nd=1)
    ndim = inputs.ndim

    if np.isscalar(min_distance):
        min_distance = (int(min_distance), ) * ndim

    ndim_peak = len(min_distance)

    if ndim_peak > ndim:
        raise ValueError('min distance must have length equal or smaller then the input rank')
    if axis is not None and len(axis) != ndim_peak:
        raise ValueError('axis must have length equal to min_distance length')

    min_distance = tuple(int(mm) for mm in min_distance)
    if any(mm < 0 for mm in min_distance):
        raise ValueError('min distance must have positive and larger then zero integers')

    axis = valid_axis(ndim, axis, ndim_peak)

    if invert:
        inputs = invert_values(inputs)

    if threshold is None:
        threshold = 0.5 * np.max(inputs)

    padding_mode = ctype_border_mode(padding_mode)

    need_transpose, transpose_forward, transpose_back = axis_transpose_to_last(ndim, axis, default_nd=ndim_peak)

    if need_transpose:
        inputs = inputs.transpose(transpose_forward)

    if ndim_peak == ndim:
        output = c_pycv.peak_nonmaximum_suppression(
            inputs, min_distance, threshold, padding_mode, constant_value
        )
    else:
        output, _ = get_output(np.int64, inputs)
        _iter = iterators.ArrayIteratorSlice(inputs.shape, ndim_peak)
        prev_max = 0
        for slc in _iter:
            output[slc] = c_pycv.peak_nonmaximum_suppression(
                inputs[slc], min_distance, threshold, padding_mode, constant_value
            )
            output[slc] += prev_max
            prev_max = np.max(output[slc])

    if need_transpose:
        output = output.transpose(transpose_back)

    return output


########################################################################################################################
