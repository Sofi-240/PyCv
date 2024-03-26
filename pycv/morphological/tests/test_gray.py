import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
import pytest
from pycv._lib.array_api.dtypes import INTEGER_DTYPES, FLOAT_DTYPES, BOOL_DTYPES
from pycv import morphological as morph

TYPES = INTEGER_DTYPES | FLOAT_DTYPES | BOOL_DTYPES


########################################################################################################################


class TestGrayErosion:
    @pytest.mark.parametrize('input_dtype', TYPES)
    def test_case1(self, input_dtype):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 3, 1, 3, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=input_dtype
        )
        strel = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=input_dtype
        )

        output = morph.gray_erosion(inputs, strel=strel)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('output_dtype', TYPES)
    def test_case2(self, output_dtype):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 3, 1, 3, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        strel = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=output_dtype
        )
        output = np.zeros_like(inputs, dtype=output_dtype)
        output = morph.gray_erosion(inputs, strel=strel, output=output)
        assert_array_almost_equal(output, expected)

    def test_case3(self):
        assert_raises(TypeError, morph.gray_erosion, np.zeros([]))

    def test_case4(self):
        inputs = np.ones([1], np.uint8)
        assert_array_almost_equal(
            morph.gray_erosion(inputs),
            [0]
        )

    def test_case5(self):
        inputs = np.ones([1], np.uint8)
        assert_array_almost_equal(
            morph.gray_erosion(inputs, border_val=1),
            [1]
        )

    def test_case6(self):
        inputs = np.ones([3], np.uint8)
        assert_array_almost_equal(
            morph.gray_erosion(inputs),
            [0, 1, 0]
        )

    def test_case7(self):
        inputs = np.ones([3], np.uint8)
        assert_array_almost_equal(
            morph.gray_erosion(inputs, border_val=1),
            [1, 1, 1]
        )

    def test_case8(self):
        inputs = np.ones([5], np.uint8)
        inputs[2] = 0
        assert_array_almost_equal(
            morph.gray_erosion(inputs),
            [0, 0, 0, 0, 0]
        )

    def test_case9(self):
        inputs = np.ones([5], np.uint8)
        inputs[2] = 0
        assert_array_almost_equal(
            morph.gray_erosion(inputs, border_val=1),
            [1, 0, 0, 0, 1]
        )

    def test_case10(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 2, 1, 2, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = np.ones((3, 1), bool)

        expected = np.array(
            [[0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_erosion(inputs, strel=se, border_val=1),
            expected
        )

    def test_case11(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 2, 1, 2, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = np.ones((3, 1), bool)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_erosion(inputs, strel=se, border_val=0),
            expected
        )

    def test_case12(self):
        inputs = np.ones([5], np.uint8)
        inputs[2] = 0
        se = np.array([1, 0, 1], bool)
        assert_array_almost_equal(
            morph.gray_erosion(inputs, strel=se, border_val=1),
            [1, 0, 1, 0, 1]
        )

    def test_case13(self):
        inputs = np.ones([5], np.uint8)
        inputs[2] = 0
        se = np.array([1, 0, 1], bool)
        assert_array_almost_equal(
            morph.gray_erosion(inputs, strel=se, border_val=0),
            [0, 0, 1, 0, 0]
        )

    def test_case14(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 2, 0, 2, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = morph.Strel.DEFAULT_STREL(2, 1)
        se[1, 1] = 0

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 2, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_erosion(inputs, strel=se, border_val=0),
            expected
        )

    def test_case15(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = np.ones((2, 2), bool)
        se[1, 0] = 0
        center = (1, 1)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 1, 1, 1],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_erosion(inputs, offset=center, strel=se, border_val=0),
            expected
        )

    def test_case16(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = morph.Strel.DEFAULT_STREL(2, 1)
        center = (1, 1)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_erosion(inputs, offset=center, strel=se, border_val=0),
            expected
        )

        assert_array_almost_equal(
            morph.gray_erosion(inputs, offset=center, strel=se, border_val=1),
            expected
        )

    def test_case17(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )
        se = morph.Strel.DEFAULT_STREL(2, 1).astype(np.uint8)
        center = (1, 1)

        assert_array_almost_equal(
            morph.gray_erosion(inputs, offset=center, strel=se, border_val=0),
            np.zeros_like(inputs)
        )

        assert_array_almost_equal(
            morph.gray_erosion(inputs, offset=center, strel=se, border_val=1),
            np.zeros_like(inputs)
        )

    def test_case18(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 2, 1, 2, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = morph.Strel.DEFAULT_STREL(2, 1).astype(np.uint8)
        se[1, 1] = 0
        center = (1, 1)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_erosion(inputs, offset=center, strel=se, border_val=0),
            expected
        )

        assert_array_almost_equal(
            morph.gray_erosion(inputs, offset=center, strel=se, border_val=1),
            expected
        )


class TestGrayDilation:
    @pytest.mark.parametrize('input_dtype', TYPES)
    def test_case1(self, input_dtype):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 3, 1, 3, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=input_dtype
        )
        strel = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.array(
            [[0, 1, 2, 1, 0],
             [1, 3, 2, 3, 1],
             [3, 3, 3, 3, 3],
             [1, 3, 2, 3, 1],
             [0, 1, 2, 1, 0]],
            dtype=input_dtype
        )

        output = morph.gray_dilation(inputs, strel=strel)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('output_dtype', TYPES)
    def test_case2(self, output_dtype):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 3, 1, 3, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        strel = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.array(
            [[0, 1, 2, 1, 0],
             [1, 3, 2, 3, 1],
             [3, 3, 3, 3, 3],
             [1, 3, 2, 3, 1],
             [0, 1, 2, 1, 0]],
            dtype=output_dtype
        )
        output = np.zeros_like(inputs, dtype=output_dtype)
        output = morph.gray_dilation(inputs, strel=strel, output=output)
        assert_array_almost_equal(output, expected)

    def test_case3(self):
        assert_raises(TypeError, morph.gray_dilation, np.zeros([]))

    def test_case4(self):
        inputs = np.zeros([1], np.uint8)
        assert_array_almost_equal(
            morph.gray_dilation(inputs),
            [0]
        )

    def test_case5(self):
        inputs = np.zeros([1], np.uint8)
        assert_array_almost_equal(
            morph.gray_dilation(inputs, border_val=1),
            [1]
        )

    def test_case6(self):
        inputs = np.zeros([3], np.uint8)
        assert_array_almost_equal(
            morph.gray_dilation(inputs),
            [0, 0, 0]
        )

    def test_case7(self):
        inputs = np.zeros([3], np.uint8)
        assert_array_almost_equal(
            morph.gray_dilation(inputs, border_val=1),
            [1, 0, 1]
        )

    def test_case8(self):
        inputs = np.zeros([5], np.uint8)
        inputs[2] = 1
        assert_array_almost_equal(
            morph.gray_dilation(inputs),
            [0, 1, 1, 1, 0]
        )

    def test_case9(self):
        inputs = np.zeros([5], np.uint8)
        inputs[2] = 1
        assert_array_almost_equal(
            morph.gray_dilation(inputs, border_val=1),
            [1, 1, 1, 1, 1]
        )

    def test_case10(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 2, 1, 2, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = np.ones((3, 1), bool)

        expected = np.array(
            [[0, 1, 2, 1, 0],
             [1, 2, 2, 2, 1],
             [1, 2, 2, 2, 1],
             [1, 2, 2, 2, 1],
             [0, 1, 2, 1, 0]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_dilation(inputs, strel=se, border_val=0),
            expected
        )

    def test_case11(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 2, 1, 2, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = np.ones((3, 1), bool)

        expected = np.array(
            [[1, 1, 2, 1, 1],
             [1, 2, 2, 2, 1],
             [1, 2, 2, 2, 1],
             [1, 2, 2, 2, 1],
             [1, 1, 2, 1, 1]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_dilation(inputs, strel=se, border_val=1),
            expected
        )

    def test_case12(self):
        inputs = np.zeros([5], np.uint8)
        inputs[2] = 1
        se = np.array([1, 0, 1], bool)
        assert_array_almost_equal(
            morph.gray_dilation(inputs, strel=se, border_val=1),
            [1, 1, 0, 1, 1]
        )

    def test_case13(self):
        inputs = np.zeros([5], np.uint8)
        inputs[2] = 1
        se = np.array([1, 0, 1], bool)
        assert_array_almost_equal(
            morph.gray_dilation(inputs, strel=se, border_val=0),
            [0, 1, 0, 1, 0]
        )

    def test_case14(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 2, 1, 0],
             [1, 2, 0, 2, 1],
             [0, 1, 2, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = morph.Strel.DEFAULT_STREL(2, 1)
        se[1, 1] = 0

        expected = np.array(
            [[0, 1, 2, 1, 0],
             [1, 2, 1, 2, 1],
             [2, 1, 2, 1, 2],
             [1, 2, 1, 2, 1],
             [0, 1, 2, 1, 0]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_dilation(inputs, strel=se, border_val=0),
            expected
        )

    def test_case15(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 2],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )
        se = np.ones((2, 2), bool)
        se[1, 0] = 0
        center = (1, 1)

        expected = np.array(
            [[0, 0, 1, 1, 0],
             [0, 1, 1, 1, 2],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1],
             [0, 0, 1, 1, 1]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_dilation(inputs, offset=center, strel=se, border_val=0),
            expected
        )

    def test_case16(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )
        se = morph.Strel.DEFAULT_STREL(2, 1)
        center = (1, 1)

        expected = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_dilation(inputs, offset=center, strel=se, border_val=0),
            expected
        )

        assert_array_almost_equal(
            morph.gray_dilation(inputs, offset=center, strel=se, border_val=1),
            np.ones_like(inputs)
        )

    def test_case17(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )
        se = morph.Strel.DEFAULT_STREL(2, 1).astype(np.uint8)
        center = (1, 1)

        expected = np.array(
            [[1, 1, 2, 1, 1],
             [1, 2, 2, 2, 1],
             [2, 2, 2, 2, 2],
             [1, 2, 2, 2, 1],
             [1, 1, 2, 1, 1]],
            dtype=np.uint8
        )

        assert_array_almost_equal(
            morph.gray_dilation(inputs, offset=center, strel=se, border_val=0),
            expected
        )

    def test_case18(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        ) * 255
        se = morph.Strel.DEFAULT_STREL(2, 1).astype(np.uint8)
        center = (1, 1)

        expected = np.array(
            [[1, 1, 2, 1, 1],
             [1, 2, 2, 2, 1],
             [2, 2, 2, 2, 2],
             [1, 2, 2, 2, 1],
             [1, 1, 2, 1, 1]],
            dtype=np.uint8
        )
        expected[expected == 2] = 255

        assert_array_almost_equal(
            morph.gray_dilation(inputs, offset=center, strel=se, border_val=0),
            expected
        )


########################################################################################################################
