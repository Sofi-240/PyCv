from _debug_utils.im_load import load_image, load_defualt_binary_image
# from _debug_utils.im_viz import show_collection
import numpy as np
from pycv.colors import rgb2gray
from pycv.segmentation import im_threshold
from pycv.transform import bilinear_resize, nearest_neighbour_resize
from pycv.morphological import binary_edge, region_fill, im_label
from pycv._lib.core import ops

# rng = np.random.default_rng()
# inputs = np.zeros((256, 256))
# inputs[64:-64, 64:-64] = 1
# inputs += 0.2 * rng.random(inputs.shape)

# inputs = load_image('lena.jpg')
# show_collection([inputs, output_b, output_nn], 1, 3)

# inputs = load_image('coins.png')
# bin_image, th = im_threshold(
#     inputs, 'kapur',
# )
# bin_image = bin_image[85:, :]

# n_labels, labels = im_label(bin_image, connectivity=1)

# show_collection([inputs, bin_image, labels], 1, 3)

values = np.zeros((15, 15), np.uint8)
values[3:6, 3:6] = 100
values[6:8, 3:6] = 101
values[10:14, 10:14] = 255

n_labels, labels = im_label(values, connectivity=1, rng_mapping_method='linear', mod_value=16)
# values = bin_image

# traverser = np.zeros((values.size, ), np.int64)
# parent = np.zeros(values.shape, np.int64)
#
# ops.build_max_tree(values, traverser, parent)

# show_collection([inputs, output], 1, 2)