import numpy as np
from .._lib._src_py.utils import ctype_convex_hull_mode
from .._lib.array_api.regulator import np_compliance
from .._lib.array_api.dtypes import as_binary_array
from .._lib._src_py.pycv_minsc import draw
from pycv._lib._src.c_pycv import CConvexHull
from typing import Iterable

__all__ = [
    "ConvexHull"
]


########################################################################################################################

class ConvexHull(CConvexHull):
    def __init__(
            self,
            image: np.ndarray | None = None,
            points: np.ndarray | None = None,
            mask: np.ndarray | None = None,
            image_shape: tuple | None = None,
            method: str = 'graham'
    ):
        if image is None and points is None:
            raise ValueError('one of the parameters (image or points) need to be given')
        if image is not None:
            image = np_compliance(np.asarray(image, order='C'), 'image', _check_finite=True)
            if image.ndim != 2:
                raise RuntimeError('convex hull currently supported just for 2D arrays')
            if mask is not None:
                mask = np.asarray(mask, order='C')
                if mask.shape != image.shape:
                    raise ValueError('mask shape need to be same as image shape')
                mask = as_binary_array(mask)
                mask[image == 0] = 0
            else:
                mask = image != 0
            points = np.stack(np.where(mask), axis=-1).astype(np.int64)
            image_shape = image.shape
        else:
            points = np_compliance(np.asarray(points, order='C'), 'points', _check_finite=True)
            if points.ndim != 2:
                raise ValueError('points need to have 2 dimensions (N points, nd)')
            points = points.astype(np.int64)

        method = ctype_convex_hull_mode(method)
        super().__init__(points, method)
        self._image_shape = image_shape or np.amax(points, axis=0) + 1

    def __repr__(self):
        return f'{self.__class__.__name__}: ndim={self.ndim}, n_vertices={self.n_vertices}'

    def to_image(self, image_shape: tuple | None = None) -> np.ndarray:
        image_shape = image_shape or self._image_shape
        if len(image_shape) != self.ndim:
            raise ValueError('image shape size need to be equal to convex ndim')
        return super().to_image(image_shape)

    def query_point(self, points: Iterable) -> np.ndarray:
        points = np.asarray(points, dtype=np.int64)
        if points.ndim != 2:
            raise ValueError('points need to have 2 dimensions (N points, nd)')
        if points.shape[1] != self.ndim:
            raise ValueError('points.shape[1] need to be equal to convex ndim')
        return np.array(super().query_point(points), dtype=bool)

    def edge_image(self, image_shape: tuple | None = None) -> np.ndarray:
        image_shape = image_shape or self._image_shape
        if len(image_shape) != self.ndim:
            raise ValueError('image shape size need to be equal to convex ndim')
        if any(s < si for s, si in zip(image_shape, self._image_shape)):
            raise ValueError('image shape is to small for convex vertices')
        output = np.zeros(image_shape, bool)

        for i in range(1, self.n_vertices):
            point1 = tuple(int(p) for p in self.points[self.vertices[i - 1]])
            point2 = tuple(int(p) for p in self.points[self.vertices[i]])
            output[draw('line', point1=point1, point2=point2)] = 1

        point1 = tuple(int(p) for p in self.points[self.vertices[-1]])
        point2 = tuple(int(p) for p in self.points[self.vertices[0]])
        output[draw('line', point1=point1, point2=point2)] = 1

        return output

########################################################################################################################

