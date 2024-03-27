import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
import pytest
from pycv._lib.array_api.dtypes import INTEGER_DTYPES, FLOAT_DTYPES, BOOL_DTYPES
from pycv import morphological as morph

TYPES = INTEGER_DTYPES | FLOAT_DTYPES | BOOL_DTYPES


########################################################################################################################

class TestLabeling:

    def test_case1(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=bool
        )

        connectivity = 1
        expected = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0],
             [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
             [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
             [0, 4, 4, 4, 0, 3, 0, 0, 0, 0, 0],
             [0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.int64
        )

        n, output = morph.im_label(inputs, connectivity)
        assert_array_almost_equal(output, expected)

    def test_case2(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=bool
        )

        connectivity = 2
        expected = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0],
             [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
             [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
             [0, 3, 3, 3, 0, 3, 0, 0, 0, 0, 0],
             [0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.int64
        )

        n, output = morph.im_label(inputs, connectivity)
        assert_array_almost_equal(output, expected)

    def test_case3(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 60, 60, 60, 0, 0, 80, 80, 80, 0, 0],
             [0, 60, 60, 60, 0, 0, 80, 80, 80, 0, 0],
             [0, 60, 60, 60, 0, 0, 80, 80, 80, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 110, 110, 110, 110, 110, 110, 0, 0, 0],
             [0, 0, 0, 110, 110, 110, 110, 0, 0, 0, 0],
             [0, 0, 0, 0, 110, 110, 110, 0, 0, 0, 0],
             [0, 40, 40, 40, 0, 110, 0, 0, 0, 0, 0],
             [0, 40, 40, 40, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.uint8
        )
        connectivity = 2
        expected = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0],
             [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
             [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
             [0, 4, 4, 4, 0, 3, 0, 0, 0, 0, 0],
             [0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.uint8
        )
        n, output = morph.im_label(inputs, connectivity, rng_mapping_method='linear')
        assert_array_almost_equal(output, expected)

    def test_case4(self):
        assert_raises(TypeError, morph.im_label, np.zeros([]))

    def test_case5(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 60, 60, 60, 0, 0, 80, 80, 80, 0, 0],
             [0, 60, 60, 60, 0, 0, 80, 80, 80, 0, 0],
             [0, 60, 60, 60, 0, 0, 80, 80, 80, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 110, 110, 110, 110, 110, 110, 0, 0, 0],
             [0, 0, 0, 110, 110, 110, 110, 0, 0, 0, 0],
             [0, 0, 0, 0, 110, 110, 110, 0, 0, 0, 0],
             [0, 40, 40, 40, 0, 110, 0, 0, 1, 1, 0],
             [0, 40, 40, 40, 0, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.uint8
        )
        connectivity = 1
        expected = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0],
             [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
             [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
             [0, 4, 4, 4, 0, 3, 0, 0, 5, 5, 0],
             [0, 4, 4, 4, 0, 0, 0, 6, 0, 0, 7],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        n, output = morph.im_label(inputs, connectivity, rng_mapping_method='linear')
        assert_array_almost_equal(output, expected)

    def test_case6(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0],
             [0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
             [0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 0, 0, 2, 2, 0],
             [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.uint8
        )
        connectivity = 1
        expected = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0],
             [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
             [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
             [0, 4, 4, 4, 0, 5, 0, 0, 6, 6, 0],
             [0, 4, 4, 4, 0, 0, 0, 7, 0, 0, 8],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        n, output = morph.im_label(inputs, connectivity, rng_mapping_method='linear', mod_value=1)
        assert_array_almost_equal(output, expected)

    def test_case7(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0],
             [0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
             [0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 0, 0, 2, 2, 0],
             [0, 1, 1, 1, 0, 0, 0, 2, 0, 0, 2],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.uint8
        )
        connectivity = 2
        expected = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0],
             [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
             [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
             [0, 4, 4, 4, 0, 5, 0, 0, 6, 6, 0],
             [0, 4, 4, 4, 0, 0, 0, 6, 0, 0, 6],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        n, output = morph.im_label(inputs, connectivity, rng_mapping_method='linear', mod_value=1)
        assert_array_almost_equal(output, expected)

    def test_case8(self):
        inputs = np.array(
            [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
             [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
             [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
             [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
             [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]],
            dtype=bool
        )
        connectivity = 1
        expected = np.array(
            [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0, 2, 0, 1, 1, 1, 1],
             [1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1],
             [1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1],
             [1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1],
             [1, 1, 1, 1, 0, 2, 0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]],
            dtype=np.uint8
        )

        n, output = morph.im_label(inputs, connectivity)
        assert_array_almost_equal(output, expected)

        n, output = morph.im_label(inputs, 2)
        assert_array_almost_equal(output, inputs)

    def test_case9(self):
        inputs = np.array(
            [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
             [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
             [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
             [1, 0, 1, 0, 1, 4, 1, 0, 1, 0, 1],
             [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
             [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
             [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]],
            dtype=np.uint8
        )
        connectivity = 2
        expected = np.array(
            [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 2, 2, 2, 0, 0, 1, 0],
             [1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1],
             [1, 0, 2, 0, 0, 3, 0, 0, 2, 0, 1],
             [1, 0, 2, 0, 3, 4, 3, 0, 2, 0, 1],
             [1, 0, 2, 0, 0, 3, 0, 0, 2, 0, 1],
             [1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1],
             [0, 1, 0, 0, 2, 2, 2, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]],
            dtype=np.uint8
        )

        n, output = morph.im_label(inputs, connectivity)
        assert_array_almost_equal(output, expected)

    def test_case10(self):
        inputs = np.array(
            [1, 1, 1, 0, 1, 1, 1],
            dtype=bool
        )
        connectivity = 1
        expected = np.array(
            [1, 1, 1, 0, 2, 2, 2],
            dtype=np.int64
        )

        n, output = morph.im_label(inputs, connectivity)
        assert_array_almost_equal(output, expected)

    def test_case11(self):
        inputs = np.array(
            [1, 0, 1],
            dtype=bool
        )
        connectivity = 1
        expected = np.array(
            [1, 0, 2],
            dtype=np.int64
        )

        n, output = morph.im_label(inputs, connectivity)
        assert_array_almost_equal(output, expected)

    def test_case12(self):
        inputs = np.array(
            [[1], [0], [1]],
            dtype=bool
        )
        expected = np.array(
            [[1], [0], [2]],
            dtype=np.int64
        )

        n, output = morph.im_label(inputs, 1)
        assert_array_almost_equal(output, expected)
        n, output = morph.im_label(inputs, 2)
        assert_array_almost_equal(output, expected)

    def test_case13(self):
        inputs = np.zeros((7, 15, 15), bool)
        expected = np.zeros((7, 15, 15), np.int64)

        sp = morph.Strel.SPHERE(2)

        inputs[1:-1, 3 - 2: 3 + 3, 3 - 2: 3 + 3] = sp
        inputs[1:-1, 10 - 2: 10 + 3, 10 - 2: 10 + 3] = sp

        expected[1:-1, 3 - 2: 3 + 3, 3 - 2: 3 + 3][sp] = 1
        expected[1:-1, 10 - 2: 10 + 3, 10 - 2: 10 + 3][sp] = 2

        n, output = morph.im_label(inputs, 2)
        assert_array_almost_equal(output, expected)


class TestFillRegion:
    def test_case1(self):
        expected = morph.Strel.DIAMOND(5)
        inputs = morph.binary_edge(expected)

        strel = morph.Strel.DEFAULT_STREL(2, 1)
        point = tuple(s // 2 for s in inputs.shape)

        output = morph.region_fill(inputs, seed_point=point, strel=strel)
        assert_array_almost_equal(output, expected)

    def test_case2(self):
        expected = morph.Strel.DIAMOND(5).astype(np.uint8)
        inputs = morph.binary_edge(expected)

        strel = morph.Strel.DEFAULT_STREL(2, 1)
        point = tuple(s // 2 for s in inputs.shape)

        output = morph.region_fill(inputs, seed_point=point, strel=strel)
        assert_array_almost_equal(output, expected)

    def test_case3(self):
        output = morph.region_fill(np.array([1, 0, 0, 0, 1], bool), seed_point=(2, ), )
        assert_array_almost_equal(output, np.array([1, 1, 1, 1, 1]))

    def test_case4(self):
        output = morph.region_fill(np.array([1, 0, 0, 0, 1], bool), seed_point=(2, ), strel=np.ones([1], bool))
        assert_array_almost_equal(output, np.array([1, 0, 0, 0, 1]))

    def test_case5(self):
        inputs = np.zeros((7, 7, 7), bool)
        output = morph.region_fill(inputs, seed_point=(0, 0, 0))
        assert_array_almost_equal(output, np.ones_like(inputs))

    def test_case6(self):
        expected = np.zeros((13, ) * 3, bool)
        expected[1:-1, 1:-1, 1:-1] = morph.Strel.OCTAHEDRON(5)
        inputs = morph.binary_edge(expected)
        output = morph.region_fill(inputs, seed_point=(6, 6, 6))
        assert_array_almost_equal(output, expected)