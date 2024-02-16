import numpy as np
import abc

__all__ = [
    "ArrayIteratorSlice",
    "PointsIteratorSlice"
]


########################################################################################################################

class _IteratorBase(abc.ABC):
    def __init__(self):
        self._i = 0

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i == self.iter_size:
            self._i = 0
            raise StopIteration
        out = self._next()
        self._i += 1
        return out

    @property
    @abc.abstractmethod
    def iter_size(self):
        """
        Return the iterations size
        """

    @abc.abstractmethod
    def _next(self):
        """
        Return next element
        """


########################################################################################################################

class ArrayIteratorSlice(_IteratorBase):
    def __init__(self, array_shape: tuple, out_ndim: int):
        super().__init__()
        self._d_ndim = len(array_shape) - out_ndim
        self._iter_size = int(np.prod(array_shape[:-out_ndim]))
        self._dims_m1 = np.array(array_shape[:-out_ndim], np.int64) - 1
        self._coordinates = [0] * self._d_ndim

    @property
    def iter_size(self):
        return self._iter_size

    def _next(self):
        out = tuple(self._coordinates)
        for ii in range(self._d_ndim - 1, -1, -1):
            if self._coordinates[ii] < self._dims_m1[ii]:
                self._coordinates[ii] += 1
                break
            else:
                self._coordinates[ii] = 0
        return out


class PointsIteratorSlice(_IteratorBase):
    def __init__(self, points: np.ndarray, out_ndim: int):
        super().__init__()
        self._d_ndim = points.shape[1] - out_ndim
        shape = tuple(np.max(points[:, c]) + 1 for c in range(points.shape[1]))
        strides = shape[1:] + (1,)
        strides = np.cumprod(strides[::-1])[::-1]
        self._map = (points[:, :-out_ndim] * strides[:-out_ndim]).sum(axis=1)
        self._map_val = np.unique(self._map)
        self._iter_size = self._map_val.shape[0]
        self._ii = 0

    @property
    def iter_size(self):
        return self._iter_size

    def _next(self):
        out = (self._map == self._map_val[self._ii], slice(self._d_ndim, None, None))
        self._ii = (self._ii + 1) % self.iter_size
        return out
