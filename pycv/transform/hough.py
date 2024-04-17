import numpy as np
from .._lib._src_py import pycv_transform
from .._lib.array_api.shapes import atleast_nd
from ..features.peaks import _peaks_nonmaximum_suppression
from ..dsa import KDtree

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
    """
    Apply the Hough transform algorithm to detect lines in an image.

    Parameters:
        inputs (numpy.ndarray): The input image array.
        theta (numpy.ndarray or None, optional): The array of line angles (in radians). If None, the angles will be automatically generated. Defaults to None.

    Returns:
        tuple: A tuple containing the Hough transform output array, parameters (angles), and distances.
    """
    return pycv_transform.hough_transform('line', inputs, params=theta)


def hough_circle(
        inputs: np.ndarray,
        radius: np.ndarray | int,
) -> np.ndarray:
    """
    Apply the Hough transform algorithm to detect circles in an image.

    Parameters:
        inputs (numpy.ndarray): The input image array.
        radius (numpy.ndarray or int): The array of circle radii or a single integer radius to detect circles of fixed radius.

    Returns:
        numpy.ndarray: The Hough transform output array representing detected circles.

    Notes:
        - This function is a convenience wrapper around the hough_transform function with mode set to 'circle'.
    """
    return pycv_transform.hough_transform('circle', inputs, params=radius)


def hough_probabilistic_line(
        inputs: np.ndarray,
        theta: np.ndarray | None = None,
        threshold: int = 10,
        line_length: int = 50,
        line_gap: int = 8,
) -> np.ndarray:
    """
    Apply the probabilistic Hough transform algorithm to detect lines in an image.

    Parameters:
        inputs (numpy.ndarray): The input image array.
        theta (numpy.ndarray or None): The array of theta values for line angles.
            If None, theta values will be generated automatically.
        threshold (int): The minimum number of votes (intersections) required to detect a line.
        line_length (int): The minimum length of a line segment to be considered.
        line_gap (int): The maximum gap between line segments that are allowed to be connected.

    Returns:
        numpy.ndarray: The Hough transform output array representing detected lines.
    """
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
    """
    Find peaks in a Hough space for detecting lines.

    Parameters:
        h_space (numpy.ndarray): The Hough space array.
        theta (numpy.ndarray): The array of theta values corresponding to the Hough space.
        distances (numpy.ndarray): The array of distances corresponding to the Hough space.
        n_peaks (int): The number of peaks to return. Use -1 to return all peaks.
        min_distance_delta (int): The minimum distance delta between peaks.
        min_theta_delta (int): The minimum theta delta between peaks.
        threshold (float or None): The minimum value a peak must have to be considered.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing arrays representing
        the values of Hough space, theta, and distances for the detected peaks.

    """
    h_space = np.asarray(h_space)
    h_space = atleast_nd(h_space, 2, raise_err=True)

    if h_space.shape[-2] != distances.shape[0] or h_space.shape[-1] != theta.shape[0]:
        raise ValueError('h_space has invalid shape')

    min_distance = (min_distance_delta, min_theta_delta)
    peaks_mask = _peaks_nonmaximum_suppression(
        h_space, min_distance=min_distance, threshold=threshold, padding_mode='constant', num_peaks=n_peaks
    )
    peaks = np.where(peaks_mask)

    return h_space[peaks], theta[peaks[1]], distances[peaks[0]]


def hough_circle_peak(
        h_space: np.ndarray,
        radius: np.ndarray | int,
        n_peaks: int = -1,
        min_y_distance: int = 7,
        min_x_distance: int = 7,
        threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peaks in a Hough space for detecting circles.

    Parameters:
        h_space (numpy.ndarray): The Hough space array.
        radius (numpy.ndarray or int): The array of radius values or a single radius value.
        n_peaks (int): The number of peaks to return. Use -1 to return all peaks.
        min_y_distance (int): The minimum y distance delta between peaks.
        min_x_distance (int): The minimum x distance delta between peaks.
        threshold (float or None): The minimum value a peak must have to be considered.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing arrays representing the values of
        Hough space, radius, and peaks center for the detected circles.

    """
    h_space = np.asarray(h_space)
    h_space = atleast_nd(h_space, 3, raise_err=True)

    if h_space.shape[-3] != radius.shape[0]:
        raise ValueError('h_space has invalid shape')

    if h_space.shape[0] == 1:
        min_distance = (0, min_y_distance, min_x_distance)
    else:
        min_distance = (1, min_y_distance, min_x_distance)
    peaks_mask = _peaks_nonmaximum_suppression(
        h_space, min_distance=min_distance, threshold=threshold, padding_mode='constant'
    )
    peaks = np.where(peaks_mask)
    if n_peaks == -1 or n_peaks > peaks[0].size:
        n_peaks = None
    return h_space[peaks][:n_peaks], radius[peaks[0]][:n_peaks], np.stack(peaks[1:], axis=1)[:n_peaks]
