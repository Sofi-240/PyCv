import numpy as np
from .._lib.array_api.regulator import np_compliance
from pycv._lib._src.c_pycv import CKDtree

__all__ = [
    "KDtree"
]


########################################################################################################################

class KDtree(CKDtree):
    def __init__(self, data: np.ndarray, leafsize: int = 4):
        """
        KDtree constructor.

        Constructs a KD-tree for fast nearest-neighbor queries.

        Parameters:
            data (numpy.ndarray): The input data points.
            leafsize (int, optional): The number of points at which the algorithm switches to brute-force. Defaults to 4.

        Raises:
            ValueError: If data is not of shape (n [points], m [dimension]) or leafsize is not a positive integer.

        """
        data = np_compliance(data, 'data', _check_finite=True)
        if data.ndim != 2:
            raise ValueError('data must be of shape (n [points], m [dimension])')
        data = data.astype(np.float64)
        if leafsize < 1:
            raise ValueError('leafsize must be positive integer')
        super().__init__(data, leafsize=leafsize)

    def __len__(self):
        return self.size

    def knn_query(
            self,
            points: np.ndarray,
            k: int | np.ndarray,
            pnorm: int = 2,
            distance_max: float = float('inf'),
            epsilon: float = 0
    ) -> tuple[list, list] | tuple[np.ndarray, np.ndarray]:
        """
        Perform k-nearest neighbor queries.

        Parameters:
            points (numpy.ndarray): The query points.
            k (int or numpy.ndarray): The number of nearest neighbors to search for each query point.
            pnorm (int, optional): The norm to be used. Defaults to 2.
            distance_max (float, optional): The maximum distance to consider when searching for neighbors. Defaults to 2.
            epsilon (float, optional): Approximation parameter. Defaults to 0.

        Returns:
            tuple: A tuple containing lists of distances and indices of the k-nearest neighbors for each query point.

        Raises:
            TypeError: If k is not an integer or numpy.ndarray.
            ValueError: If the shapes of points and k do not match or if an unexpected error occurs.

        """
        points = np_compliance(points, 'points', _check_finite=True)
        if points.ndim != 2 or points.shape[1] != self.m:
            raise ValueError('points must be of shape (n [points], m [dimension])')

        points = points.astype(np.float64)

        if not isinstance(k, np.ndarray):
            if not np.isscalar(k):
                raise TypeError('k need to be an integer or numpy.ndarray')
            k = np.array([k] * points.shape[0], np.int64)

        if k.ndim != 1 or k.shape[0] != points.shape[0]:
            raise ValueError('k need to be 1D array with size equal to n points')

        if points.shape[0] == 0:
            return np.array([], np.float64), np.array([], np.int64)

        output = super().knn_query(points, k, pnorm, int(pnorm == float('inf')), distance_max, epsilon)

        if output is None:
            raise RuntimeError('unexpected error in knn_query')

        dist, indices = output

        if points.shape[0] == 1:
            return np.array(dist[0], np.float64), np.array(indices[0], np.int64)

        dist = [np.array(d, np.float64) for d in dist]
        indices = [np.array(d, np.int64) for d in indices]

        return dist, indices

    def ball_point_query(
            self,
            points: np.ndarray,
            radius: int | np.ndarray,
            pnorm: int = 2,
            epsilon: float = 0
    ) -> list[np.ndarray] | np.ndarray:
        """
        Perform ball point queries.

        Parameters:
            points (numpy.ndarray): The query points.
            radius (int or numpy.ndarray): The radius within which to search for neighbors for each query point.
            pnorm (int, optional): The norm to be used. Defaults to 2.
            epsilon (float, optional): Approximation parameter. Defaults to 0.

        Returns:
            list: A list containing arrays of indices of points within the specified radius for each query point.

        Raises:
            TypeError: If radius is not an scalar or numpy.ndarray.
            ValueError: If the shapes of points and radius do not match or if an unexpected error occurs.

        """
        points = np_compliance(points, 'points', _check_finite=True)
        if points.ndim != 2 or points.shape[1] != self.m:
            raise ValueError('points must be of shape (n [points], m [dimension])')

        points = points.astype(np.float64)

        if not isinstance(radius, np.ndarray):
            if not np.isscalar(radius):
                raise TypeError('radius need to be an scalar or numpy.ndarray')
            radius = np.array([radius] * points.shape[0], np.float64)

        if radius.ndim != 1 or radius.shape[0] != points.shape[0]:
            raise ValueError('radius need to be 1D array with size equal to n points')

        radius = radius.astype(np.float64)

        if points.shape[0] == 0:
            return np.array([], np.int64)

        output = super().ball_point_query(points, radius, pnorm, int(pnorm == float('inf')), epsilon)

        if output is None:
            raise RuntimeError('unexpected error in query_ball_kdtree')

        if points.shape[0] == 1:
            return np.array(output[0], np.int64)

        output = [np.array(o, np.int64) for o in output]

        return output


########################################################################################################################
