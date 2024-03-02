import numpy as np
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib._src import c_pycv

__all__ = [
    'KDtree',
    'KDnode'
]


########################################################################################################################

class _KDnode(object):
    def __init__(self, start_index: int = 0, end_index: int = 0, level: int = 0):
        self.start_index = start_index
        self.end_index = end_index
        self.split_dim = -1
        self.split_val = 0
        self.lesser_index = -1
        self.higher_index = -1
        self.lesser = None
        self.higher = None
        self.level = level

    @property
    def children(self):
        return self.end_index - self.start_index


########################################################################################################################


class KDnode(_KDnode):
    def __init__(self, **kwargs):
        super().__init__()
        for attr, val in kwargs.items():
            if attr in self.__dict__:
                setattr(self, attr, val)
        if self.children < 0:
            raise ValueError(f'end_index cannot be smaller then start_index')
        self.data = None
        self.indices = None


class KDtree(object):
    def __init__(self, data: np.ndarray, leafsize: int = 4):
        data = np_compliance(data, 'data', _check_finite=True)
        if data.ndim != 2:
            raise ValueError('data must be of shape (n [points], m [dimension])')
        self.n = data.shape[0]
        self.m = data.shape[1]
        data = data.astype(np.float64)
        self.data = data
        if leafsize < 1:
            raise ValueError('leafsize must be positive integer')
        self.leafsize = leafsize
        self.dims_min = np.min(data, axis=0)
        self.dims_max = np.max(data, axis=0)
        self.indices = np.arange(self.n, dtype=np.int64)

        self.tree_list = []

        if self.n:
            nodes = c_pycv.build_kdtree(
                self.data,
                self.dims_min,
                self.dims_max,
                self.indices,
                self.leafsize
            )
            if nodes is None:
                raise RuntimeError('unexpected error init_kdtree')
        else:
            nodes = []
        self.tree_list.extend(KDnode(**nodes[i]) for i in range(len(nodes)))
        self.size = len(self.tree_list)
        self.tree = self.tree_list[0] if self.size else None
        self.__post_init(self.tree)

    def __len__(self):
        return self.size

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def __post_init(self, node: KDnode):
        if not node:
            return

        node.indices = self.indices[node.start_index:node.end_index]
        node.data = self.data[node.indices]

        if node.split_dim == -1:
            node.lesser_index = -1
            node.higher_index = -1
            node.split_val = 0
            return

        node.lesser = self.tree_list[node.lesser_index]
        node.higher = self.tree_list[node.higher_index]

        self.__post_init(node.lesser)
        self.__post_init(node.higher)

    def query_knn(
            self,
            points: np.ndarray,
            k: int | np.ndarray,
            pnorm: int = 2,
            distance_max: float = float('inf'),
            epsilon: float = 0
    ) -> tuple[list, list] | tuple[np.ndarray, np.ndarray]:
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

        output = c_pycv.query_kdtree(self, points, k, pnorm, int(pnorm == float('inf')), distance_max, epsilon)
        if output is None:
            raise RuntimeError('unexpected error in query_kdtree')

        dist, indices, slc = output

        if points.shape[0] == 1:
            return dist[slc[0]:slc[1]], indices[slc[0]:slc[1]]

        dist_out = []
        indices_out = []

        for i in range(points.shape[0]):
            dist_out.append(dist[slc[i]:slc[i + 1]])
            indices_out.append(indices[slc[i]:slc[i + 1]])

        return dist_out, indices_out

    def query_ball_point(
            self,
            points: np.ndarray,
            radius: int | np.ndarray,
            pnorm: int = 2,
            epsilon: float = 0
    ) -> list[np.ndarray] | np.ndarray:
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

        output = c_pycv.query_ball_kdtree(self, points, radius, pnorm, int(pnorm == float('inf')), epsilon)
        if output is None:
            raise RuntimeError('unexpected error in query_ball_kdtree')

        indices, slc = output

        if points.shape[0] == 1:
            return indices[slc[0]:slc[1]]

        indices_out = []

        for i in range(points.shape[0]):
            indices_out.append(indices[slc[i]:slc[i + 1]])

        return indices_out


def _fill_tree_split(output: np.ndarray, node: KDnode, bound1: list | None = None, bound2: list | None = None):
    if not node or node.split_dim == -1:
        return
    if bound1 is None or bound2 is None:
        bound1 = [0] * output.ndim
        bound2 = list(output.shape)

    split_dim = node.split_dim
    split_point = int(node.split_val)

    slc = tuple()
    for dim in range(len(bound1)):
        if dim == split_dim:
            slc += (split_point,)
            continue
        slc += (slice(bound1[dim], bound2[dim]),)

    output[slc] = np.max(output) + 1

    lesser_b1 = bound1[:]
    lesser_b2 = [bound2[dim] if dim != split_dim else split_point for dim in range(len(bound1))]

    higher_b1 = [bound1[dim] if dim != split_dim else split_point + 1 for dim in range(len(bound1))]
    higher_b2 = bound2[:]

    _fill_tree_split(output, node.lesser, lesser_b1, lesser_b2)
    _fill_tree_split(output, node.higher, higher_b1, higher_b2)
