import numpy as np
from .._lib._src_py import pycv_peaks
from ..filters import local_max_filter, local_min_filter
from .._lib._src_py.utils import axis_transpose_to_last, valid_axis
from .._lib.array_api import iterators
from ..dsa import KDtree

__all__ = [
    'find_peaks'
]


########################################################################################################################

def _peaks_nonmaximum_suppression(
        inputs: np.ndarray,
        min_distance: tuple | int = 1,
        threshold: float | None = None,
        num_peaks: int = -1,
        axis: tuple | None = None,
        padding_mode: str = 'constant',
        constant_value: float = 0,
        invert: bool = False,
) -> np.ndarray:
    """
    Apply non-maximum suppression to detect peaks in the input array.

    Parameters:
        inputs : np.ndarray
            Input array of any dimension.
        min_distance : tuple | int
            Minimum distance between detected peaks. If an integer is provided, it will be
            used as the minimum distance for all dimensions.
        threshold : float | None, optional
            Threshold value for peak detection. If None, defaults to 0.5 times the maximum
            value of the input array.
        num_peaks: int
            The maximum number of peaks. If -1 then all the peaks found are returned.
        axis : tuple | None, optional
            Axes along which peaks are detected. If None, peaks are detected over all dimensions.
        padding_mode : str, optional
            Padding mode for handling edges. Defaults to 'constant'. Valid options are
            {'constant', 'edge', 'symmetric', 'reflect'}.
        constant_value : float, optional
            Constant value used for padding if padding_mode is 'constant'.
        invert : bool, optional
            If True, invert the values of the input array before peak detection.
    Returns:
        np.ndarray
            An array that includes the identified peaks numbered according to the value of the peak from highest to lowest.

    """
    peaks_mask = pycv_peaks.peak_nonmaximum_suppression(
        inputs, min_distance, threshold=threshold, axis=axis,
        padding_mode=padding_mode, constant_value=constant_value,
        invert=invert
    )
    if num_peaks > 0:
        peaks_mask[peaks_mask > num_peaks] = 0
    return peaks_mask


########################################################################################################################

def find_peaks(
        inputs: np.ndarray,
        min_distance: int = 1,
        footprint: np.ndarray | None = None,
        threshold: float | None = None,
        num_peaks: int = -1,
        axis: tuple | None = None,
        padding_mode: str = 'constant',
        constant_value: float = 0,
        invert: bool = False,
        pnorm: int = 2,
) -> np.ndarray:
    inputs = np.asarray(inputs)
    ndim = inputs.ndim

    ndim_peak = len(axis) if axis is not None else ndim
    k_distance = (min_distance * 2 + 1,) * ndim_peak

    axis = valid_axis(inputs.ndim, axis, ndim_peak)

    need_transpose, transpose_forward, transpose_back = axis_transpose_to_last(ndim, axis, default_nd=ndim_peak)

    if need_transpose:
        inputs = inputs.transpose(transpose_forward)

    if invert:
        mask = inputs == local_min_filter(
            inputs,
            kernel_size=k_distance,
            footprint=footprint,
            padding_mode=padding_mode,
            constant_value=constant_value,
        )
    else:
        mask = inputs == local_max_filter(
            inputs,
            kernel_size=k_distance,
            footprint=footprint,
            padding_mode=padding_mode,
            constant_value=constant_value,
        )

    if threshold is None:
        threshold = 0.5 * np.max(inputs)

    _iter = iterators.ArrayIteratorSlice(inputs.shape, ndim_peak)

    for slc in _iter:
        m = mask[slc]
        m &= (inputs[slc] > threshold) if not invert else (inputs[slc] < threshold)
        c = m.nonzero()
        v = inputs[slc][c]

        i = np.argsort(-v if not invert else v)

        p = np.stack(tuple(_c[i] for _c in c), axis=-1)
        tree = KDtree(p, leafsize=1)

        query = tree.ball_point_query(p, radius=min_distance, pnorm=pnorm)
        c = tuple(_c.squeeze() for _c in np.split(p, ndim_peak, axis=-1))

        for i, q in enumerate(query):
            if q.size > 1 and m[tuple(_c[i] for _c in c)]:
                del_p = tuple(np.squeeze(pp) for pp in np.split(p[q[1:], :], 2, axis=-1))
                m[del_p] = 0

        if 0 < num_peaks < p.shape[0]:
            c = m.nonzero()
            v = inputs[slc][c]
            i = np.argsort(-v if not invert else v)

            del_p = tuple(_c[i[num_peaks:]] for _c in c)
            m[del_p] = 0

    if need_transpose:
        mask = mask.transpose(transpose_back)

    return mask

########################################################################################################################
