import numpy as np
from numpy.testing import assert_array_almost_equal
from pycv._lib.core import ops
from pycv._lib.array_api.shapes import output_shape
from pycv._lib.filters_support.kernel_utils import color_mapping_range


class TestFilters_C(object):
    def test_convolve01(self):
        inputs = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]])
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        offset = tuple(s // 2 for s in kernel.shape)
        output = np.zeros(output_shape(inputs.shape, kernel.shape), inputs.dtype)
        ops.convolve(inputs, kernel, output, offset)
        assert_array_almost_equal([[40, 45, 50, 55]], output)

    def test_convolve02(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        kernel = np.array([[1, 0], [0, 1]])
        offset = (0, 0)
        output = np.zeros(output_shape(inputs.shape, kernel.shape), inputs.dtype)
        ops.convolve(inputs, kernel, output, offset)
        assert_array_almost_equal([[8, 10, 12, 14]], output)

    def test_convolve03(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        kernel = np.array([[1, 0, 1]])
        offset = (0, 1)
        output = np.zeros(output_shape(inputs.shape, kernel.shape), inputs.dtype)
        ops.convolve(inputs, kernel, output, offset)
        assert_array_almost_equal([[4, 6, 8], [14, 16, 18]], output)

    def test_convolve04(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        kernel = np.array([[1], [0], [1]])
        offset = (1, 0)
        output = np.zeros(output_shape(inputs.shape, kernel.shape), inputs.dtype)
        ops.convolve(inputs, kernel, output, offset)
        assert_array_almost_equal([[12, 14, 16, 18, 20]], output)

    def test_convolve05(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        kernel = np.array([[1]])
        offset = (0, 0)
        output = np.zeros(output_shape(inputs.shape, kernel.shape), inputs.dtype)
        ops.convolve(inputs, kernel, output, offset)
        assert_array_almost_equal(inputs, output)

    def test_rank_filter01(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        offset = (1, 1)

        rank_median = np.sum(footprint) // 2
        rank_min = 0
        rank_max = np.sum(footprint) - 1

        output_median = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_median, rank_median, offset)
        assert_array_almost_equal([[7, 8, 9]], output_median)

        output_min = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_min, rank_min, offset)
        assert_array_almost_equal([[2, 3, 4]], output_min)

        output_max = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_max, rank_max, offset)
        assert_array_almost_equal([[12, 13, 14]], output_max)

    def test_rank_filter02(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        footprint = np.array([[1], [1], [1]])
        offset = (1, 0)

        rank_median = footprint.size // 2
        rank_min = 0
        rank_max = footprint.size - 1

        output_median = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_median, rank_median, offset)
        assert_array_almost_equal([[6, 7, 8, 9, 10]], output_median)

        output_min = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_min, rank_min, offset)
        assert_array_almost_equal([[1, 2, 3, 4, 5]], output_min)

        output_max = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_max, rank_max, offset)
        assert_array_almost_equal([[11, 12, 13, 14, 15]], output_max)

    def test_rank_filter03(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        footprint = np.array([[1, 1, 1]])
        offset = (0, 1)

        rank_median = footprint.size // 2
        rank_min = 0
        rank_max = footprint.size - 1

        output_median = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_median, rank_median, offset)
        assert_array_almost_equal([[2, 3, 4], [7, 8, 9], [12, 13, 14]], output_median)

        output_min = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_min, rank_min, offset)
        assert_array_almost_equal([[1, 2, 3], [6, 7, 8], [11, 12, 13]], output_min)

        output_max = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_max, rank_max, offset)
        assert_array_almost_equal([[3, 4, 5], [8, 9, 10], [13, 14, 15]], output_max)

    def test_rank_filter04(self):
        inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        footprint = np.array([[1, 0], [1, 1]])
        offset = (0, 0)

        rank_median = np.sum(footprint) // 2
        rank_min = 0
        rank_max = np.sum(footprint) - 1

        output_median = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_median, rank_median, offset)
        assert_array_almost_equal([[6, 7, 8, 9]], output_median)

        output_min = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_min, rank_min, offset)
        assert_array_almost_equal([[1, 2, 3, 4]], output_min)

        output_max = np.zeros(output_shape(inputs.shape, footprint.shape), inputs.dtype)
        ops.rank_filter(inputs, footprint, output_max, rank_max, offset)
        assert_array_almost_equal([[7, 8, 9, 10]], output_max)


class TestMorphology_C(object):
    def test_erosion_dilation01(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        strel = np.ones((3, 3), inputs.dtype)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        output = np.zeros(inputs.shape, bool)
        offset = tuple(s // 2 for s in strel.shape)

        ops.binary_erosion(inputs, strel, output, offset, 1, None, 0)
        assert_array_almost_equal(expected, output)

        output = np.zeros(inputs.shape, bool)
        ops.binary_erosion(expected, strel, output, offset, 1, None, 1)
        assert_array_almost_equal(inputs, output)

    def test_erosion_dilation02(self):
        inputs = np.array(
            [[1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1]],
            dtype=bool
        )

        strel = np.ones((3, 3), inputs.dtype)

        mask = np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        expected = np.array(
            [[1, 1, 1, 1, 1],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 1, 1, 1]],
            dtype=bool
        )

        output = np.zeros(inputs.shape, bool)
        offset = tuple(s // 2 for s in strel.shape)

        ops.binary_erosion(inputs, strel, output, offset, 1, mask, 0)
        assert_array_almost_equal(expected, output)

        output = np.zeros(inputs.shape, bool)
        ops.binary_erosion(expected, strel, output, offset, 1, mask, 1)
        assert_array_almost_equal(inputs, output)

    def test_binary_region_fill01(self):
        inputs = np.array(
            [[1, 1, 1, 1, 1],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 1, 1, 1]],
            dtype=bool
        )

        strel = np.ones((3, 3), inputs.dtype)

        expected = np.ones_like(inputs)

        offset = tuple(s // 2 for s in strel.shape)

        ops.binary_region_fill(inputs, strel, (2, 2), offset)
        assert_array_almost_equal(inputs, expected)

    def test_gray_erosion01(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 2, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        strel = np.ones((3, 3), bool)
        offset = tuple(s // 2 for s in strel.shape)

        output = np.zeros_like(inputs)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        ops.erosion(inputs, strel, None, output, offset, None, 0)
        assert_array_almost_equal(output, expected)

    def test_gray_erosion02(self):
        inputs = np.array(
            [[1, 1, 1, 1, 1],
             [1, 2, 2, 2, 1],
             [1, 2, 3, 2, 1],
             [1, 2, 2, 2, 1],
             [1, 1, 1, 1, 1]],
            dtype=np.uint8
        )

        flat_strel = np.ones((3, 3), bool)
        non_flat_strel = np.ones((3, 3), np.uint8)
        offset = tuple(s // 2 for s in flat_strel.shape)

        output = np.zeros_like(inputs)

        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        ops.erosion(inputs, flat_strel, non_flat_strel, output, offset, None, 0)
        assert_array_almost_equal(output, expected)

    def test_gray_erosion03(self):
        inputs = np.array(
            [[1, 1, 1, 1, 1],
             [1, 2, 2, 2, 1],
             [1, 2, 3, 2, 1],
             [1, 2, 2, 2, 1],
             [1, 1, 1, 1, 1]],
            dtype=np.uint8
        )

        flat_strel = np.ones((3, 3), bool)
        non_flat_strel = np.ones((3, 3), np.uint8)
        offset = tuple(s // 2 for s in flat_strel.shape)

        output = np.zeros_like(inputs)

        mask = np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        expected = np.array(
            [[1, 1, 1, 1, 1],
             [1, 0, 0, 0, 1],
             [1, 0, 1, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 1, 1, 1]],
            dtype=np.uint8
        )

        ops.erosion(inputs, flat_strel, non_flat_strel, output, offset, mask, 0)
        assert_array_almost_equal(output, expected)

    def test_gray_dilation01(self):
        inputs = np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 2, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8
        )

        strel = np.ones((3, 3), inputs.dtype)
        offset = tuple(s // 2 for s in strel.shape)

        output = np.zeros_like(inputs)

        expected = np.array(
            [[1, 1, 1, 1, 1],
             [1, 2, 2, 2, 1],
             [1, 2, 2, 2, 1],
             [1, 2, 2, 2, 1],
             [1, 1, 1, 1, 1]],
            dtype=np.uint8
        )

        ops.dilation(inputs, strel, None, output, offset, None, 255)
        assert_array_almost_equal(output, expected)

    def test_gray_dilation02(self):
        inputs = np.array(
            [[1, 1, 1, 1, 1],
             [1, 2, 2, 2, 1],
             [1, 2, 3, 2, 1],
             [1, 2, 2, 2, 1],
             [1, 1, 1, 1, 1]],
            dtype=np.uint8
        )

        flat_strel = np.ones((3, 3), bool)
        non_flat_strel = np.ones((3, 3), np.uint8)
        offset = tuple(s // 2 for s in flat_strel.shape)

        output = np.zeros_like(inputs)

        expected = np.array(
            [[3, 3, 3, 3, 3],
             [3, 4, 4, 4, 3],
             [3, 4, 4, 4, 3],
             [3, 4, 4, 4, 3],
             [3, 3, 3, 3, 3]],
            dtype=np.uint8
        )

        ops.dilation(inputs, flat_strel, non_flat_strel, output, offset, None, 255)
        assert_array_almost_equal(output, expected)

    def test_gray_dilation03(self):
        inputs = np.array(
            [[1, 1, 1, 1, 1],
             [1, 2, 2, 2, 1],
             [1, 2, 3, 2, 1],
             [1, 2, 2, 2, 1],
             [1, 1, 1, 1, 1]],
            dtype=np.uint8
        )

        flat_strel = np.ones((3, 3), bool)
        non_flat_strel = np.ones((3, 3), np.uint8)
        offset = tuple(s // 2 for s in flat_strel.shape)

        output = np.zeros_like(inputs)

        mask = np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]],
            dtype=bool
        )

        expected = np.array(
            [[1, 1, 1, 1, 1],
             [1, 4, 4, 4, 1],
             [1, 4, 4, 4, 1],
             [1, 4, 4, 4, 1],
             [1, 1, 1, 1, 1]],
            dtype=np.uint8
        )

        ops.dilation(inputs, flat_strel, non_flat_strel, output, offset, mask, 255)
        assert_array_almost_equal(output, expected)

    def test_binary_labeling01(self):
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

        output = np.zeros_like(inputs, dtype=np.uint8)

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

        ops.labeling(inputs, connectivity, None, output)

        assert_array_almost_equal(output, expected)

    def test_binary_labeling02(self):
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

        output = np.zeros_like(inputs, dtype=np.uint8)

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
            dtype=np.uint8
        )

        ops.labeling(inputs, connectivity, None, output)

        assert_array_almost_equal(output, expected)

    def test_gray_labeling01(self):
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

        output = np.zeros_like(inputs, dtype=np.uint8)
        values_map = color_mapping_range(inputs, 'linear')

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

        ops.labeling(inputs, connectivity, values_map, output)

        assert_array_almost_equal(output, expected)