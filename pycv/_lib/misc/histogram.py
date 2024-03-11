import numpy as np
from pycv._lib.array_api.dtypes import get_dtype_info

__all__ = [
    'Histogram',
    'bin_count',
    'histogram',
]


########################################################################################################################

def _get_bins_outer_edges(image: np.ndarray, bins_range: tuple | None = None) -> tuple:
    if bins_range is None:
        bins_range = (np.amin(image), np.amax(image))
    elif image.size == 0:
        bins_range = (0, 1)
    else:
        if len(bins_range) != 2:
            raise ValueError(f'Bins range len need to be 2 (min, max)')
        if bins_range[0] > bins_range[1]:
            raise ValueError(f'max need to be larger than min in bins_range parameter')

    lo, hi = bins_range
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f'Bins range {bins_range} is not finite')
    if lo == hi:
        lo -= 0.5
        hi += 0.5

    return lo, hi


def _get_bins_edge(image: np.ndarray, bins: int | np.ndarray, bins_range: tuple | None = None) -> np.ndarray:
    n_bins = None
    bins_edge = None
    if np.isscalar(bins):
        n_bins = int(bins)
        if n_bins < 1:
            raise ValueError('bins as an integer mast be positive')
    else:
        bins_edge = np.asarray(bins)
        _info = get_dtype_info(bins_edge.dtype)  # check if dtype in supported types
        if bins_edge.ndim != 1:
            raise ValueError('bins as array must be 1d')
        if np.any(bins_edge[:-1] > bins_edge[1:]):
            raise ValueError('bins as array must increase monotonically')

    if n_bins is not None:
        lo, hi = _get_bins_outer_edges(image, bins_range)
        bin_type = np.result_type(lo, hi, image)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)
        bins_edge = np.linspace(lo, hi, n_bins + 1, endpoint=True, dtype=bin_type)

    return bins_edge


def _get_bins_center(bins_edge: np.ndarray) -> np.ndarray:
    return (bins_edge[:-1] + bins_edge[1:]) / 2


########################################################################################################################

class Histogram(object):
    def __init__(self, counts: np.ndarray, bins: np.ndarray):
        self.counts = counts
        self.bins = bins

    def __repr__(self):
        _out = f"{self.__class__.__name__}: " \
               f"counts = \n{np.array2string(self.counts, separator=', ')}"
        return _out

    def __array__(self):
        return self.counts

    @property
    def shape(self) -> tuple:
        return self.counts.shape

    @property
    def n_bins(self) -> int:
        return self.shape[1]

    @property
    def normalize(self) -> np.ndarray:
        c = self.counts
        if c.dtype.kind != 'f':
            dtype = np.result_type(c, float)
            c = c.astype(dtype)
        return c / np.sum(c, axis=1)

    def cdf(self) -> np.ndarray:
        c = self.counts
        if c.dtype.kind != 'f':
            dtype = np.result_type(c, float)
            c = c.astype(dtype)
        _cdf = c.cumsum(axis=1)
        _cdf /= np.reshape(_cdf[:, -1], (-1, 1))
        return _cdf


########################################################################################################################


def _bin_count(
        image: np.ndarray,
        bins_range: tuple | None = None,
        channels: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError('Image dtype need to be uint or int for bin_count')

    if channels is not None:
        if channels >= image.ndim:
            raise ValueError(f'channels {channels} is out of range for array with {image.ndim} dimensions')
        channels = channels % image.ndim if channels < 0 else channels

    lo, hi = _get_bins_outer_edges(image, bins_range)

    if lo < 0:
        dtype = np.result_type(lo, hi - lo)
        if image.dtype != dtype:
            image = image.astype(dtype)
        image -= lo

    bin_type = np.result_type(lo, hi, image)
    if np.issubdtype(bin_type, np.integer):
        bin_type = np.result_type(bin_type, float)

    bins_edge = np.arange(lo, hi + 1).astype(bin_type)
    min_length = hi - min(lo, 0) + 1
    s = max(lo, 0)

    if channels is None:
        h = np.bincount(image[(image >= lo) & (image <= hi)].ravel(), minlength=min_length)[s:]
        h = h.reshape((1, -1))
    else:
        h = []
        h.extend(
            np.bincount(
                im[(im >= lo) & (im <= hi)].ravel(), minlength=min_length
            )[s:] for im in np.split(image, image.shape[channels], axis=channels)
        )
        h = np.stack(h, axis=0)

    dtype = np.result_type(h, float)
    h = h.astype(dtype)
    return h, bins_edge


def _histogram(
        image: np.ndarray,
        bins: int | np.ndarray | None = None,
        bins_range: tuple | None = None,
        channels: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if np.issubdtype(image.dtype, np.integer) and bins is None:
        return _bin_count(image, bins_range, channels)

    if channels is not None:
        if channels >= image.ndim:
            raise ValueError(f'channels {channels} is out of range for array with {image.ndim} dimensions')
        channels = channels % image.ndim if channels < 0 else channels

    bins = bins if bins is not None else 10
    bins_edge = _get_bins_edge(image, bins, bins_range)

    if channels is None:
        h, _ = np.histogram(image.ravel(), bins_edge)
        h = h.reshape((1, -1))
    else:
        h = []
        h.extend(
            np.histogram(im.ravel(), bins_edge)[0] for im in np.split(image, image.shape[channels], axis=channels)
        )
        h = np.stack(h, axis=0)

    dtype = np.result_type(h, float)
    h = h.astype(dtype)

    return h, _get_bins_center(bins_edge)


########################################################################################################################


def bin_count(
        image: np.ndarray,
        bins_range: tuple | None = None,
        channels: int | None = None
) -> Histogram:
    h, bins = _bin_count(image, bins_range, channels)
    return Histogram(h, bins)


def histogram(
        image: np.ndarray,
        bins: int | np.ndarray | None = None,
        bins_range: tuple | None = None,
        channels: int | None = None
) -> Histogram:
    h, bins = _histogram(image, bins, bins_range, channels)
    return Histogram(h, bins)


########################################################################################################################
