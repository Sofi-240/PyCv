import numpy as np
from pycv._lib._src_py import pycv_transform, pycv_measure
from pycv._lib.array_api.shapes import atleast_nd
from pycv._lib._src_py.kdtree import KDtree

__all__ = [
    'hough_line',
    'hough_circle',
    'hough_probabilistic_line',
    'hough_line_peak',
    'hough_circle_peak'
]


########################################################################################################################

def hough_line(
        inputs: np.ndarray,
        theta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pycv_transform.hough_transform('line', inputs, params=theta)


def hough_circle(
        inputs: np.ndarray,
        radius: np.ndarray | int,
) -> np.ndarray:
    return pycv_transform.hough_transform('circle', inputs, params=radius)


def hough_probabilistic_line(
        inputs: np.ndarray,
        theta: np.ndarray | None = None,
        threshold: int = 10,
        line_length: int = 50,
        line_gap: int = 8,
) -> np.ndarray:
    return pycv_transform.hough_transform(
        'pp_line', inputs, params=theta, threshold=threshold, line_length=line_length, line_gap=line_gap
    )


########################################################################################################################

def hough_line_peak(
        h_space: np.ndarray,
        theta: np.ndarray,
        distances: np.ndarray,
        n_peaks: int = -1,
        min_distance_delta: int = 7,
        min_theta_delta: int = 7,
        threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_space = np.asarray(h_space)
    h_space = atleast_nd(h_space, 2, raise_err=True)

    if h_space.shape[-2] != distances.shape[0] or h_space.shape[-1] != theta.shape[0]:
        raise ValueError('h_space has invalid shape')

    min_distance = (min_distance_delta, min_theta_delta)
    peaks_mask = pycv_measure.find_object_peaks(
        h_space, min_distance=min_distance, threshold=threshold, padding_mode='constant',
    )
    peaks = np.where(peaks_mask)

    if 0 <= n_peaks < peaks[0].shape[0]:
        sorted_ = np.argsort(h_space[peaks])[::-1]
        peaks = tuple(p[sorted_] for p in peaks)

    return h_space[peaks], theta[peaks[1]], distances[peaks[0]]


def hough_circle_peak(
        h_space: np.ndarray,
        radius: np.ndarray | int,
        n_peaks: int = -1,
        min_y_distance: int = 7,
        min_x_distance: int = 7,
        threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_space = np.asarray(h_space)
    h_space = atleast_nd(h_space, 3, raise_err=True)

    if h_space.shape[-3] != radius.shape[0]:
        raise ValueError('h_space has invalid shape')

    min_distance = (min_y_distance, min_x_distance)
    peaks_mask = pycv_measure.find_object_peaks(
        h_space, min_distance=min_distance, threshold=threshold, padding_mode='constant',
    )
    peaks = np.where(peaks_mask)
    if 0 <= n_peaks < peaks[0].shape[0]:
        sorted_ = np.argsort(h_space[peaks])[::-1]
        peaks = tuple(p[sorted_] for p in peaks)

    peaks_radius = radius[peaks[0]]
    peaks_cc = np.stack(peaks[1:], axis=1)
    peaks_h = h_space[peaks]

    tree = KDtree(peaks_cc, 1)
    query_nn = tree.query_ball_point(peaks_cc, np.hypot(min_y_distance, min_x_distance))

    mask = np.ones_like(peaks_radius, bool)

    for ii, nn in enumerate(query_nn):
        if mask[ii]:
            for jj in nn:
                if jj != ii:
                    mask[jj] = 0

    return peaks_h[mask], peaks_radius[mask]
