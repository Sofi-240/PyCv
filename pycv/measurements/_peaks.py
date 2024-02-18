import numpy as np
from pycv._lib.array_api.dtypes import as_binary_array
from pycv._lib.array_api.array_pad import pad, get_padding_width
from pycv._lib._src_py import pycv_morphology
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib._src_py.utils import fix_kernel_shape, valid_axis

__all__ = [
    'find_object_peak'
]

########################################################################################################################


def find_object_peak(
        inputs: np.ndarray,
        min_distance: tuple | None = None,
        footprint: np.ndarray | None = None,
        axis: int | tuple | None = None,
        threshold: float = 0,
        num_peaks: int | None = None,
        connectivity: int = 1,
        padding_mode: str = 'constant',
        constant_value: float | None = 0.0
) -> np.ndarray:
    inputs = np_compliance(inputs, 'inputs', _check_finite=True)
    ndim = inputs.ndim

    if footprint is not None:
        footprint = np_compliance(footprint, 'footprint', _check_finite=True)
        min_distance = tuple(s // 2 for s in footprint.shape)
    elif min_distance is None:
        min_distance = (1, ) * ndim
        footprint = np.ones((3, ) * ndim, bool)
    else:
        footprint = np.ones(tuple(min(m * 2 + 1, s - 1) for m, s in zip(min_distance, inputs.shape)), bool)
        min_distance = tuple(s // 2 for s in footprint.shape)

    if footprint.ndim > ndim:
        raise ValueError('footprint number of dimensions is larger then the inputs number of dimensions')
    footprint = as_binary_array(footprint)
    axis = valid_axis(ndim, axis, footprint.ndim)

    if len(axis) != footprint.ndim:
        raise ValueError('footprint ndim dont match axis len')
    kernel_shape = fix_kernel_shape(footprint.shape, axis, ndim)
    footprint = np.reshape(footprint, kernel_shape)

    if padding_mode not in ['constant', 'valid']:
        inputs = pad(inputs, get_padding_width(footprint.shape), mode=padding_mode)
        constant_value = 0
        padding_mode = 'valid'

    max_inp = pycv_morphology.gray_ero_or_dil(1, inputs, footprint, offset=tuple(s // 2 for s in footprint.shape), border_val=constant_value)

    if padding_mode == 'valid':
        pw = get_padding_width(footprint.shape)
        max_inp = max_inp[tuple(slice(s[0], sh - s[1]) for (s, sh) in zip(pw, inputs.shape))]

    if threshold is None:
        threshold = np.min(inputs)

    is_max = np.where((max_inp > threshold) & (max_inp == inputs), True, False)
    peaks_val = np.where(is_max, max_inp, 0)
    n_labels, labels = pycv_morphology.labeling(is_max, connectivity)

    peaks = [l + 1 for l in range(n_labels)]
    peaks.sort(key=lambda l: peaks_val[labels == l][0], reverse=True)

    grid = np.mgrid[tuple(slice(-mm, mm + 1) for mm in min_distance)]
    grid = grid.astype(np.int64)
    out = []

    for l_peak in peaks:
        peak_center = np.round(np.stack(np.where(labels == l_peak), axis=1).mean(axis=0)).astype(np.int64)
        if peaks_val[tuple(np.split(peak_center, ndim, axis=0))] <= threshold:
            continue
        offsets = np.reshape(peak_center, (-1, ) + (1, ) * ndim) + grid
        offsets = [f.ravel() for f in np.split(offsets, ndim, axis=0)]
        mask = np.ones_like(offsets[0], bool)
        for j in range(ndim):
            mask &= (offsets[j] >= 0) & (offsets[j] < inputs.shape[j])

        peaks_val[tuple(f[mask] for f in offsets)] = 0
        out.append(peak_center)
        if len(out) == num_peaks:
            break

    if not out:
        return np.zeros((0, ndim), np.int64)
    out = np.stack(out, axis=0)

    return out

















