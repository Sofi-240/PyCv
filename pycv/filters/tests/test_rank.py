import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
import pytest
from pycv._lib.array_api.dtypes import INTEGER_DTYPES, FLOAT_DTYPES, BOOL_DTYPES
from pycv._lib._src_py import pycv_filters as flt

TYPES = INTEGER_DTYPES | FLOAT_DTYPES | BOOL_DTYPES


########################################################################################################################


class TestRankFilter:
    @pytest.mark.parametrize('input_dtype', TYPES - BOOL_DTYPES)
    def test_case1(self, input_dtype):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]], input_dtype
        )
        footprint = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], bool
        )
        expected = np.array(
            [[0, 0, 0],
             [0, 2, 0],
             [0, 0, 0]]
        )
        output = flt.rank_filter(inputs, footprint, rank=1)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('output_dtype', TYPES - BOOL_DTYPES)
    def test_case2(self, output_dtype):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        footprint = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], bool
        )
        expected = np.array(
            [[0, 0, 0],
             [0, 2, 0],
             [0, 0, 0]], output_dtype
        )
        output = np.zeros_like(inputs, dtype=output_dtype)
        output = flt.rank_filter(inputs, footprint, rank=1, output=output)
        assert_array_almost_equal(output, expected)

    def test_case3(self):
        assert_raises(TypeError, flt.rank_filter, np.zeros([]), np.zeros([]), 1)

    def test_case4(self):
        assert_raises(TypeError, flt.rank_filter, np.zeros([3]), np.zeros([]), 1)

    def test_case5(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        footprint = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], bool
        )
        expected = np.array(
            [[1, 1, 2],
             [1, 2, 3],
             [4, 5, 6]]
        )
        output = flt.rank_filter(inputs, footprint, rank=1, padding_mode='symmetric')
        assert_array_almost_equal(output, expected)

    def test_case6(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        footprint = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], bool
        )
        expected = np.array(
            [[4, 5, 6],
             [7, 8, 9],
             [8, 9, 9]]
        )
        output = flt.rank_filter(inputs, footprint, rank=5, padding_mode='symmetric')
        assert_array_almost_equal(output, expected)

    def test_case7(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        footprint = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]], bool
        )
        expected = np.array(
            [[2, 3, 3],
             [5, 5, 5],
             [7, 7, 8]]
        )
        output = flt.rank_filter(inputs, footprint, rank=3, padding_mode='reflect')
        assert_array_almost_equal(output, expected)
        output = flt.rank_filter(inputs, footprint, rank=3, padding_mode='symmetric')
        assert_array_almost_equal(output, inputs)