import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
import pytest
from pycv._lib.array_api.dtypes import INTEGER_DTYPES, FLOAT_DTYPES, BOOL_DTYPES
from pycv import morphological as morph

TYPES = INTEGER_DTYPES | FLOAT_DTYPES | BOOL_DTYPES


########################################################################################################################

class TestBinaryErosion:

    @pytest.mark.parametrize('input_dtype', TYPES)
    def test_case1(self, input_dtype):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
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

        output = morph.binary_erosion(inputs, strel=strel)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('output_dtype', TYPES)
    def test_case2(self, output_dtype):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )
        strel = morph.Strel.DEFAULT_STREL(2, 1)

        output = np.zeros_like(inputs, dtype=output_dtype)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=output_dtype
        )

        output = morph.binary_erosion(inputs, strel=strel, output=output)
        assert_array_almost_equal(output, expected)

    def test_case3(self):
        assert_raises(TypeError, morph.binary_erosion, np.zeros([]))

    def test_case4(self):
        inputs = np.ones([1], bool)
        assert_array_almost_equal(
            morph.binary_erosion(inputs),
            [0]
        )

    def test_case5(self):
        inputs = np.ones([1], bool)
        assert_array_almost_equal(
            morph.binary_erosion(inputs, border_val=1),
            [1]
        )

    def test_case6(self):
        inputs = np.ones([3], bool)
        assert_array_almost_equal(
            morph.binary_erosion(inputs),
            [0, 1, 0]
        )

    def test_case7(self):
        inputs = np.ones([3], bool)
        assert_array_almost_equal(
            morph.binary_erosion(inputs, border_val=1),
            [1, 1, 1]
        )

    def test_case8(self):
        inputs = np.ones([5], bool)
        inputs[2] = 0
        assert_array_almost_equal(
            morph.binary_erosion(inputs),
            [0, 0, 0, 0, 0]
        )

    def test_case9(self):
        inputs = np.ones([5], bool)
        inputs[2] = 0
        assert_array_almost_equal(
            morph.binary_erosion(inputs, border_val=1),
            [1, 0, 0, 0, 1]
        )

    def test_case10(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )
        se = np.ones((3, 1), bool)

        expected = np.array(
            [[0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=se, border_val=1),
            expected
        )

    def test_case11(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )
        se = np.ones((1, 3), bool)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [1, 1, 1, 1, 1],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=se, border_val=1),
            expected
        )

    def test_case12(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )
        se = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=se, border_val=0),
            expected
        )

        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=se, border_val=1),
            expected
        )

    def test_case13(self):
        inputs = np.ones([5], bool)
        inputs[2] = 0
        se = np.array([1, 0, 1], bool)
        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=se, border_val=1),
            [1, 0, 1, 0, 1]
        )

    def test_case14(self):
        inputs = np.ones([5], bool)
        inputs[2] = 0
        se = np.array([1, 0, 1], bool)
        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=se, border_val=0),
            [0, 0, 1, 0, 0]
        )

    def test_case15(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 0, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )
        se = morph.Strel.DEFAULT_STREL(2, 1)
        se[1, 1] = 0

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=se, border_val=0),
            expected
        )

    def test_case16(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=bool
        )
        assert_array_almost_equal(
            morph.binary_erosion(inputs, iterations=-1),
            np.zeros_like(inputs)
        )
        assert_array_almost_equal(
            morph.binary_erosion(inputs, iterations=4),
            np.zeros_like(inputs)
        )

    def test_case17(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
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
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_erosion(inputs, offset=center, strel=se, border_val=0),
            expected
        )

    def test_case18(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )
        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=np.ones((1, 1), bool)),
            inputs
        )
        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=np.zeros((1, 1), bool)),
            inputs
        )

    def test_case19(self):
        inputs = morph.Strel.SPHERE(2)
        se_sub = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            morph.binary_erosion(inputs[i], strel=se_sub, border_val=0, output=expected[i])

        se = np.zeros((3,) * 3, bool)
        se[1] = se_sub

        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=se, border_val=0),
            expected
        )

    def test_case20(self):
        inputs = morph.Strel.SPHERE(2)
        se_sub = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            morph.binary_erosion(inputs[i], strel=se_sub, border_val=0, output=expected[i])

        se = np.reshape(se_sub, (1, 3, 3))

        assert_array_almost_equal(
            morph.binary_erosion(inputs, strel=se, border_val=0),
            expected
        )


class TestBinaryDilation:

    @pytest.mark.parametrize('input_dtype', TYPES)
    def test_case1(self, input_dtype):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=input_dtype
        )
        strel = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.array(
            [[0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0]],
            dtype=input_dtype
        )

        output = morph.binary_dilation(inputs, strel=strel)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('output_dtype', TYPES)
    def test_case2(self, output_dtype):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )
        strel = morph.Strel.DEFAULT_STREL(2, 1)

        output = np.zeros_like(inputs, dtype=output_dtype)

        expected = np.array(
            [[0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0]],
            dtype=output_dtype
        )

        output = morph.binary_dilation(inputs, strel=strel, output=output)
        assert_array_almost_equal(output, expected)

    def test_case3(self):
        assert_raises(TypeError, morph.binary_dilation, np.zeros([]))

    def test_case4(self):
        inputs = np.zeros([1], bool)
        assert_array_almost_equal(
            morph.binary_dilation(inputs),
            [0]
        )

    def test_case5(self):
        inputs = np.zeros([1], bool)
        assert_array_almost_equal(
            morph.binary_dilation(inputs, border_val=1),
            [1]
        )

    def test_case6(self):
        inputs = np.zeros([3], bool)
        assert_array_almost_equal(
            morph.binary_dilation(inputs),
            [0, 0, 0]
        )

    def test_case7(self):
        inputs = np.zeros([3], bool)
        assert_array_almost_equal(
            morph.binary_dilation(inputs, border_val=1),
            [1, 0, 1]
        )

    def test_case8(self):
        inputs = np.zeros([5], bool)
        inputs[2] = 1
        assert_array_almost_equal(
            morph.binary_dilation(inputs),
            [0, 1, 1, 1, 0]
        )

    def test_case9(self):
        inputs = np.zeros([5], bool)
        inputs[2] = 1
        assert_array_almost_equal(
            morph.binary_dilation(inputs, border_val=1),
            [1, 1, 1, 1, 1]
        )

    def test_case10(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )
        se = np.ones((3, 1), bool)

        expected = np.array(
            [[0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0]],
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=se),
            expected
        )

    def test_case11(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [1, 1, 1, 1, 1],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        se = np.ones((1, 3), bool)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=se, border_val=0),
            expected
        )

    def test_case12(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )
        se = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=se, border_val=0),
            expected
        )

    def test_case13(self):
        inputs = np.zeros([5], bool)
        inputs[2] = 1
        se = np.array([1, 0, 1], bool)
        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=se, border_val=1),
            [1, 1, 0, 1, 1]
        )

    def test_case14(self):
        inputs = np.zeros([5], bool)
        inputs[2] = 1
        se = np.array([1, 0, 1], bool)
        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=se, border_val=0),
            [0, 1, 0, 1, 0]
        )

    def test_case15(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )
        se = morph.Strel.DEFAULT_STREL(2, 1)
        se[1, 1] = 0

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=se, border_val=0),
            expected
        )

    def test_case16(self):
        expected = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=bool
        )
        inputs = np.zeros_like(expected)
        inputs[4, 4] = 1

        assert_array_almost_equal(
            morph.binary_dilation(inputs, iterations=-1),
            np.ones_like(inputs)
        )
        assert_array_almost_equal(
            morph.binary_dilation(inputs, iterations=3),
            expected
        )

    def test_case17(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 1, 1, 1],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )
        se = np.ones((2, 2), bool)
        se[1, 0] = 0
        center = (1, 1)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1],
             [0, 0, 1, 1, 1],
             [0, 0, 1, 1, 1],
             [0, 0, 0, 1, 1]],
            dtype=bool
        )

        assert_array_almost_equal(
            morph.binary_dilation(inputs, offset=center, strel=se, border_val=0),
            expected
        )

    def test_case18(self):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=bool
        )
        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=np.ones((1, 1), bool)),
            inputs
        )
        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=np.zeros((1, 1), bool)),
            inputs
        )

    def test_case19(self):
        inputs = morph.Strel.SPHERE(2)
        se_sub = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            morph.binary_dilation(inputs[i], strel=se_sub, border_val=0, output=expected[i])

        se = np.zeros((3,) * 3, bool)
        se[1] = se_sub

        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=se, border_val=0),
            expected
        )

    def test_case20(self):
        inputs = morph.Strel.SPHERE(2)
        se_sub = morph.Strel.DEFAULT_STREL(2, 1)

        expected = np.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            morph.binary_dilation(inputs[i], strel=se_sub, border_val=0, output=expected[i])

        se = np.reshape(se_sub, (1, 3, 3))

        assert_array_almost_equal(
            morph.binary_dilation(inputs, strel=se, border_val=0),
            expected
        )


class TestSkeletonize:

    @pytest.mark.parametrize('input_dtype', TYPES)
    def test_case1(self, input_dtype):
        inputs = np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=input_dtype
        )
        expected = np.zeros_like(inputs)
        expected[2, 2] = 1
        assert_array_almost_equal(
            morph.skeletonize(inputs),
            expected
        )

    def test_case2(self):
        inputs = morph.Strel.CROSS((5, 5))

        assert_array_almost_equal(
            morph.skeletonize(inputs),
            inputs
        )

    def test_case3(self):
        inputs = np.zeros((5, 5), bool)
        inputs[2, 2] = 1
        assert_array_almost_equal(
            morph.skeletonize(inputs),
            inputs
        )


########################################################################################################################
