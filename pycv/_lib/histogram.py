import numpy as np
import collections

HIST = collections.namedtuple('histogram', 'counts, bins')

__all__ = [
    'HIST',
    'get_bins_edge',
    'get_bins_range',
    'edge_to_center',
    'histogram',
    'bin_count',
    'cdf',
]


########################################################################################################################

def get_bins_range(
        image: np.ndarray,
        bins_range: tuple | None = None
) -> tuple:
    if bins_range is None:
        bins_range = (np.min(image), np.max(image))

    if len(bins_range) != 2:
        raise ValueError(f'Bins range len need to be 2 (min, max)')

    min_, max_ = bins_range
    if min_ > max_:
        raise ValueError(f'max need to be larger than min in bins_range parameter')

    if not (np.isfinite(min_) and np.isfinite(max_)):
        raise ValueError(f'Bins range {bins_range} is not finite')
    if min_ == max_:
        min_ -= 0.5
        max_ += 0.5
    return min_, max_,


def get_bins_edge(
        bins_range: tuple,
        n_bins: int = 255
) -> np.ndarray:
    if len(bins_range) != 2:
        raise ValueError(f'Bins range len need to be 2 (min, max)')
    min_, max_ = bins_range
    return np.linspace(min_, max_, n_bins + 1, endpoint=True, dtype=np.float32)


def edge_to_center(
        bins_edge: np.ndarray
) -> np.ndarray:
    bins_center = (bins_edge[1:] + bins_edge[:-1]) / 2
    return bins_center


########################################################################################################################

def bin_count(
        image: np.ndarray,
        normalize: bool = False,
        channels: int | None = None
) -> HIST:
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError('Image dtype need to be uint or int for bin_count')
    if channels is not None and channels >= image.ndim:
        raise ValueError(f'channels {channels} is out pf range for array with {image.ndim} dimensions')

    min_, max_ = get_bins_range(image)

    if min_ < 0:
        new_dtype = np.result_type(min_, max_ - min_)
        if image.dtype != new_dtype:
            image = image.astype(new_dtype)
        image -= min_
    bins = np.arange(min_, max_ + 1)

    def _bin_count(arr):
        h = np.bincount(arr.ravel(), minlength=(max_ - min(min_, 0)) + 1)
        h = h[max(min_, 0):]
        h = h.astype(np.float32)
        if normalize: h /= np.sum(h)
        return h

    if channels is None:
        hist = _bin_count(image)
    else:
        hist = []
        hist.extend(
            _bin_count(s) for s in np.split(image, image.shape[channels], axis=channels)
        )
        hist = np.stack(hist, axis=0)

    return HIST(hist, bins)


def histogram(
        image: np.ndarray,
        bins: int | np.ndarray = 255,
        bins_range: tuple | None = None,
        normalize: bool = False,
        channels: int | None = None
) -> HIST:
    if channels is not None and channels >= image.ndim:
        raise ValueError(f'channels {channels} is out pf range for array with {image.ndim} dimensions')

    if isinstance(bins, np.ndarray):
        if bins.ndim > 1:
            raise ValueError('Bins need to be a int or 1D array')
    else:
        bins_range = get_bins_range(image, bins_range)
        bins = get_bins_edge(bins_range, bins)

    def _histogram(arr):
        h, _ = np.histogram(arr.ravel(), bins)
        h = h.astype(np.float32)
        if normalize: h /= np.sum(h)
        return h

    if channels is None:
        hist = _histogram(image)
    else:
        hist = []
        hist.extend(
            _histogram(s) for s in np.split(image, image.shape[channels], axis=channels)
        )
        hist = np.stack(hist, axis=0)
    bins = edge_to_center(bins)
    return HIST(hist, bins)


########################################################################################################################


def cdf(
        image: np.ndarray,
        bins: int = 255,
        channels: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    hist_obj = histogram(image, bins, channels=channels)
    hist = hist_obj.counts

    if channels is None:
        hist_cdf = hist.cumsum()
        hist_cdf = hist_cdf.astype(np.float64)
        hist_cdf /= hist_cdf[-1]
        return hist_obj

    hist_cdf = hist.cumsum(axis=1)
    hist_cdf = hist_cdf.astype(np.float64)
    hist_cdf /= np.reshape(hist_cdf[:, -1], (-1, 1))
    return hist_obj