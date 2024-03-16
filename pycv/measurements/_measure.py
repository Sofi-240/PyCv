import numpy as np
from pycv.morphological import find_object

__all__ = [
    'Bbox',
]


########################################################################################################################

class Bbox(object):
    def __init__(self, bbox: tuple):
        if all(isinstance(b, slice) for b in bbox):
            self.bbox = [
                np.array(tuple(b.start for b in bbox), dtype=np.int64),
                np.array(tuple(b.stop - 1 for b in bbox), dtype=np.int64)
            ]
        else:
            if len(bbox) != 2:
                raise ValueError('bbox must be tuple with size of 2 (top left, bottom right)')
            self.bbox = list(np.array(b, dtype=np.int64) for b in bbox)

    def __repr__(self):
        return f'{self.__class__.__name__}: ' \
               f'{np.array2string(self.top_left, separator=", ")}, {np.array2string(self.bottom_right, separator=", ")}'

    def __call__(self, image: np.ndarray):
        if image.ndim not in (self.ndim, self.ndim + 1):
            raise ValueError(f'invalid image dimensions expected to be '
                             f'{self.ndim} or {self.ndim + 1} for color image')
        if any(b >= s for b, s in zip(self.bottom_right, image.shape)):
            raise ValueError(f'image dimensions length is to small')
        return image[self.slice]

    @property
    def ndim(self):
        return self.top_left.shape[0]

    @property
    def top_left(self) -> np.ndarray:
        return self.bbox[0]

    @property
    def bottom_right(self) -> np.ndarray:
        return self.bbox[1]

    @property
    def centroid(self) -> np.ndarray:
        return sum(self.bbox) / 2

    @property
    def centroid_point(self) -> np.ndarray:
        return (self.centroid + 0.5).astype(np.int64)

    @property
    def area(self) -> int:
        return np.prod(self.bottom_right - self.top_left + 1)

    @property
    def slice(self) -> tuple[slice]:
        return tuple(slice(s, e + 1) for s, e in zip(*self.bbox))

########################################################################################################################


