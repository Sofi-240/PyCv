import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
import pytest
from pycv._lib.array_api.dtypes import INTEGER_DTYPES, FLOAT_DTYPES, BOOL_DTYPES
from pycv._lib._src_py import pycv_filters as flt
from pycv._lib.filters_support._windows import default_binary_strel

TYPES = INTEGER_DTYPES | FLOAT_DTYPES | BOOL_DTYPES


########################################################################################################################


class TestConvolve:
    @pytest.mark.parametrize('input_dtype', TYPES - BOOL_DTYPES)
    def test_case1(self, input_dtype):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=input_dtype
        )
        kernel = default_binary_strel(2, connectivity=1).astype(np.float64)

        expected = np.array(
            [[0, 0, 1, 0, 0],
             [0, 2, 2, 2, 0],
             [1, 2, 5, 2, 1],
             [0, 2, 2, 2, 0],
             [0, 0, 1, 0, 0]],
            dtype=input_dtype
        )

        output = flt.convolve(inputs, kernel, padding_mode='constant', constant_value=0)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('kernel_dtype', TYPES - BOOL_DTYPES)
    def test_case2(self, kernel_dtype):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
        )
        kernel = default_binary_strel(2, connectivity=1).astype(kernel_dtype)

        expected = np.array(
            [[0, 0, 1, 0, 0],
             [0, 2, 2, 2, 0],
             [1, 2, 5, 2, 1],
             [0, 2, 2, 2, 0],
             [0, 0, 1, 0, 0]],
        )
        output = flt.convolve(inputs, kernel, padding_mode='constant', constant_value=0)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('output_dtype', TYPES - BOOL_DTYPES)
    def test_case3(self, output_dtype):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.float64
        )
        kernel = default_binary_strel(2, connectivity=1).astype(np.float64)

        expected = np.array(
            [[0, 0, 1, 0, 0],
             [0, 2, 2, 2, 0],
             [1, 2, 5, 2, 1],
             [0, 2, 2, 2, 0],
             [0, 0, 1, 0, 0]],
            dtype=output_dtype
        )
        output = np.zeros_like(inputs, dtype=output_dtype)

        output = flt.convolve(inputs, kernel, output=output, padding_mode='constant', constant_value=0)
        assert_array_almost_equal(output, expected)

    def test_case4(self):
        assert_raises(TypeError, flt.convolve, np.zeros([]), np.zeros([]))

    def test_case5(self):
        assert_raises(TypeError, flt.convolve, np.ones([3, 3]), np.zeros([]))

    def test_case6(self):
        inputs = np.ones([1], np.float64)
        kernel = default_binary_strel(1, connectivity=1).astype(np.float64)
        assert_array_almost_equal(
            flt.convolve(inputs, kernel, padding_mode='constant', constant_value=0),
            [1]
        )

    def test_case7(self):
        inputs = np.ones([1], np.float64)
        kernel = default_binary_strel(1, connectivity=1).astype(np.float64)
        assert_array_almost_equal(
            flt.convolve(inputs, kernel, padding_mode='constant', constant_value=1),
            [3]
        )

    def test_case8(self):
        inputs = np.array([0, 1, 0], bool)
        kernel = default_binary_strel(1, connectivity=1).astype(np.float64)
        assert_array_almost_equal(
            flt.convolve(inputs, kernel, padding_mode='constant', constant_value=1),
            [1, 1, 1]
        )
        assert_array_almost_equal(
            flt.convolve(inputs, kernel, padding_mode='constant', constant_value=0),
            [1, 1, 1]
        )

    def test_case9(self):
        inputs = np.array([1, 2])
        kernel = np.array([2])

        assert_array_almost_equal(
            flt.convolve(inputs, kernel),
            [2, 4]
        )

    def test_case10(self):
        inputs = np.arange(256).reshape(16, 16)
        kernel = np.array([2])

        assert_array_almost_equal(
            flt.convolve(inputs, kernel),
            inputs * 2
        )

    def test_case11(self):
        inputs = np.array([1, 2, 3])
        kernel = np.array([1])

        assert_array_almost_equal(
            flt.convolve(inputs, kernel),
            inputs
        )

    def test_case12(self):
        inputs = np.array([1])
        kernel = np.array([1, 1])

        assert_array_almost_equal(
            flt.convolve(inputs, kernel, offset=(0, ), padding_mode='symmetric'),
            [2]
        )

    def test_case13(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        kernel = np.array(
            [[1, 1, 1]]
        )

        expected = np.array(
            [[3, 6, 5],
             [9, 15, 11]]
        )

        assert_array_almost_equal(
            flt.convolve(inputs, kernel),
            expected
        )

    def test_case14(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        kernel = np.array(
            [[1, 1, 1]]
        )

        expected = np.array(
            [[6],
             [15]]
        )

        assert_array_almost_equal(
            flt.convolve(inputs, kernel, padding_mode='valid'),
            expected
        )

    def test_case15(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        kernel = np.array(
            [[1], [1], [1]]
        )

        expected = np.array(
            [[5, 7, 9],
             [5, 7, 9]]
        )

        assert_array_almost_equal(
            flt.convolve(inputs, kernel),
            expected
        )

    def test_case16(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        kernel = np.array(
            [[1], [1], [1]]
        )

        expected = np.array(
            [[12, 15, 18]]
        )

        assert_array_almost_equal(
            flt.convolve(inputs, kernel, padding_mode='valid'),
            expected
        )

    def test_case17(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        kernel = np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        )

        expected = np.array(
            [[45]]
        )

        assert_array_almost_equal(
            flt.convolve(inputs, kernel, padding_mode='valid'),
            expected
        )

    def test_case18(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        kernel = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]]
        )

        expected = np.array(
            [[7, 11, 11],
             [17, 25, 23],
             [19, 29, 23]]
        )

        assert_array_almost_equal(
            flt.convolve(inputs, kernel),
            expected
        )

    def test_case19(self):
        inputs = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        kernel = np.array(
            [[1, 0],
             [0, 1]]
        )

        expected = np.array(
            [[2, 3, 5],
             [5, 6, 8]]
        )

        assert_array_almost_equal(
            flt.convolve(inputs, kernel, offset=(1, 1), padding_mode='symmetric'),
            expected
        )