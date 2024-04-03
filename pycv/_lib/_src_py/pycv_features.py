import numpy as np
from ..._lib.array_api.regulator import np_compliance
from ..._lib.array_api.iterators import ArrayIteratorSlice
from pycv._lib._src import c_pycv

__all__ = [
    'gray_co_occurrence_matrix'
]


########################################################################################################################

def gray_co_occurrence_matrix(
        inputs: np.ndarray,
        distances: np.ndarray,
        angle: np.ndarray,
        levels: int | None = None,
        symmetric: bool = False,
        normalize: bool = False
) -> np.ndarray:
    inputs = np_compliance(inputs, arg_name='inputs', _check_finite=True, _check_atleast_nd=2)
    distances = np_compliance(distances, arg_name='distances', _check_finite=True)
    angle = np_compliance(angle, arg_name='angle', _check_finite=True)

    if distances.ndim != 1 or angle.ndim != 1:
        raise TypeError('distances and angle need to be 1D array')

    if np.issubdtype(inputs.dtype, np.floating):
        raise TypeError('float type is not supported inputs must be with integer dtype')
    if np.issubdtype(inputs.dtype, np.signedinteger) and np.any(inputs < 0):
        raise TypeError('negative values is not supported')

    if not levels:
        levels = 256

    if np.amax(inputs) >= levels:
        raise ValueError('the maximum inputs value need to be smaller then levels')

    iter_ = ArrayIteratorSlice(inputs.shape, out_ndim=2)

    output = []

    for slc in iter_:
        glcm = c_pycv.gray_co_occurrence_matrix(inputs[slc], distances, angle, levels)

        if symmetric:
            glcm += np.transpose(glcm, (1, 0, 2, 3))

        if normalize:
            glcm = glcm.astype(np.float64)
            s = np.sum(glcm, axis=(0, 1), keepdims=True)
            s[s == 0] = 1
            glcm /= s

        output.append(glcm)

    if len(output) == 1:
        return output[0]

    return np.stack(output)


########################################################################################################################
