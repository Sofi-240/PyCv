from numpy.testing import assert_array_almost_equal, assert_raises
import pytest
import numpy as np
from pycv._lib._src.c_pycv import CKDnode
from pycv.dsa._kdtree import KDtree


########################################################################################################################

def _valid_kdtree_split(_tree: KDtree, _node: CKDnode) -> bool:
    if _node.split_dim == -1:
        return True
    _s, _e = _node.lesser.start_index, _node.lesser.end_index
    if not all(_tree.data[_i, _node.split_dim] <= _node.split_val for _i in _tree.indices[_s: _e]):
        return False
    _s, _e = _node.higher.start_index, _node.higher.end_index
    if not all(_tree.data[_i, _node.split_dim] >= _node.split_val for _i in _tree.indices[_s: _e]):
        return False
    return _valid_kdtree_split(_tree, _node.lesser) and _valid_kdtree_split(_tree, _node.higher)


def _valid_knn_query_(_data, _points, _k, _output):
    _expected = []
    for _p in _points:
        dist = np.sqrt(np.sum((_data - _p.reshape((1, -1))) ** 2, axis=-1))
        ii = np.argsort(dist)
        _expected.append(ii[:_k])

    for _e, _o in zip(_expected, _output[1]):
        if not all(_ee == _oo for _ee, _oo in zip(sorted(_e), sorted(_o))):
            return False
    return True


########################################################################################################################

class TestKDtree:
    @pytest.mark.parametrize('ndim', [1, 2, 3, 4])
    def test_case1(self, ndim):
        points = np.random.randint(0, 512, (1000, ndim))
        tree = KDtree(points)
        assert _valid_kdtree_split(tree, tree.tree)

    @pytest.mark.parametrize('leafsize', [1, 2, 3, 4, 5, 6, 7, 8])
    def test_case2(self, leafsize):
        points = np.random.randint(0, 512, (1000, 2))
        tree = KDtree(points, leafsize=leafsize)
        assert _valid_kdtree_split(tree, tree.tree)

    def test_case3(self):
        points = np.random.randint(0, 512, (50, 2))
        tree = KDtree(points, leafsize=1)
        query_points = points[:2]
        knn = tree.knn_query(query_points, k=2, pnorm=2)
        assert _valid_knn_query_(tree.data, query_points, 2, knn)

    def test_case4(self):
        arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.int64
        )

        points = np.stack(np.where(arr), axis=-1)
        mid_point = [[8, 8]]

        tree = KDtree(points, leafsize=1)
        ball3 = tree.ball_point_query(mid_point, radius=3, pnorm=2)
        ball5 = tree.ball_point_query(mid_point, radius=6, pnorm=2)

        expected = np.zeros_like(arr)

        for ind in ball5:
            expected[tuple(p for p in points[ind])] = 1

        for ind in ball3:
            expected[tuple(p for p in points[ind])] = 3

        assert_array_almost_equal(arr, expected)