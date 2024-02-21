import numpy as np
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.array_api import iterators
from pycv._lib._src_py.utils import get_output, ctype_border_mode, axis_transpose_to_last, invert_values, valid_axis
from pycv._lib._src import c_pycv


__all__ = [
    'find_object_peaks'
]


########################################################################################################################

def find_object_peaks(
        inputs: np.ndarray,
        min_distance: tuple,
        threshold: float | None = None,
        axis: tuple | None = None,
        padding_mode: str = 'constant',
        constant_value: float = 0,
        invert: bool = False
) -> np.ndarray:
    inputs = np_compliance(inputs, 'Inputs', _check_finite=True)
    ndim = inputs.ndim
    ndim_peak = len(min_distance)

    if ndim_peak > ndim:
        raise ValueError('min distance must have length equal or smaller then the input rank')
    if axis is not None and len(axis) != ndim_peak:
        raise ValueError('axis must have length equal to min_distance length')

    min_distance = tuple(int(mm) for mm in min_distance)
    if any(mm <= 0 for mm in min_distance):
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

    output, _ = get_output(bool, inputs)

    if ndim_peak == ndim:
        c_pycv.find_object_peaks(inputs, min_distance, threshold, padding_mode, constant_value, output)
    else:
        _iter = iterators.ArrayIteratorSlice(inputs.shape, ndim_peak)
        for slc in _iter:
            c_pycv.find_object_peaks(
                inputs[slc], min_distance, threshold, padding_mode, constant_value, output[slc]
            )

    if need_transpose:
        output = output.transpose(transpose_back)
    return output


########################################################################################################################
