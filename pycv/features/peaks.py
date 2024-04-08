import numpy as np
from .._lib._src_py import pycv_peaks

__all__ = [
    'peaks_nonmaximum_suppression'
]


########################################################################################################################

def peaks_nonmaximum_suppression(
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
