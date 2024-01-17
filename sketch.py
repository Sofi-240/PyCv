import numpy as np
# from pycv._lib.core import ops
# from _debug_utils.im_load import load_image
# from _debug_utils.im_viz import show_collection
# from pycv.colors.color import rgb2gray
# from pycv._lib.filters_support.filters import canny_filter
# from pycv._lib.filters_support.kernel_utils import unraveled_offsets, ravel_offsets
from pycv._lib.array_api.array_pad import get_padding_width, pad
from pycv._lib.core_support.filters_py import convolve

inputs = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]])
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

i_p = pad(inputs, ((1, 1), (1, 1)), mode='edge')
output = convolve(i_p, kernel, None, padding_mode='valid')

output2 = convolve(inputs, kernel, None, padding_mode='edge')

print(np.all(output2 == output))

# arr = np.ravel_multi_index(np.indices((11, 11)), (11, 11))
# k = np.ones((3, 3))
#
# a_reflect = pad(arr, get_padding_width(k.shape), mode='reflect')
# a_symmetric = pad(arr, get_padding_width(k.shape), mode='symmetric')
# a_wrap = pad(arr, get_padding_width(k.shape), mode='wrap')
# a_edge = pad(arr, get_padding_width(k.shape), mode='edge')



# # rng = np.random.default_rng()
# inputs = np.zeros((256, 256))
# inputs[64:-64, 64:-64] = 1
# # inputs += 0.2 * rng.random(inputs.shape)
#
# k1 = np.ones((3, 3))
# cent1 = tuple(s // 2 for s in k1.shape)
#
# k2 = np.ones((3, 3))
# cent2 = tuple(s // 2 for s in k2.shape)
#
# c1 = unraveled_offsets(k1, cent1)
# c2 = unraveled_offsets(k2, cent2)
#
# c1 = ravel_offsets(c1, inputs.shape)
# c2 = ravel_offsets(c2, inputs.shape)
#
# c2_offsets = np.zeros_like(c2)
#
# jump1 = k1.shape[1] * (cent1[0] - cent2[0]) + (cent1[1] - cent2[1])
#
# jump2 = 2 * (cent1[1] - cent2[1]) + 1
#
# count = k2.shape[1]
#
# idx = jump1
#
# for i in range(c2_offsets.size):
#     c2_offsets[i] = c1[idx]
#     count -= 1
#     if count == 0:
#         count = k2.shape[1]
#         idx += jump2
#     else:
#         idx += 1
#
# print(np.all(c2 == c2_offsets))

# inputs = rgb2gray(load_image('lena.jpg'))

# magnitude, gy, gx = canny_filter(inputs)
# theta = np.arctan2(gy, gx) * 180 / np.pi
# theta = ((theta % 180) // 22.5)
# theta = np.select(list(theta == i for i in range(8)), [4, 1, 1, 2, 2, 3, 3, 4], 0)
#
# is_down = gy <= 0
# is_up = gy >= 0
#
# is_left = gx <= 0
# is_right = gx >= 0
#
# cond1 = (is_up & is_right) | (is_down & is_left)
# cond2 = (is_down & is_right) | (is_up & is_left)
#
# gy_abs = np.abs(gy)
# gx_abs = np.abs(gx)
#
# a = np.where(cond1 & (gy > gx), 1, 0)
# a = np.where(cond1 & (gy <= gx), 2, a)
#
# a = np.where(cond2 & (gy < gx), 3, a)
# a = np.where(cond2 & (gy >= gx), 4, a)


# min_threshold_r, max_threshold_r = 0.25, 0.20
# hys_kernel = (5, 5)
# output = np.zeros_like(inputs)
# show_collection([inputs, output], 1, 2)

