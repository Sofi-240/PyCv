from _debug_utils.im_load import load_image
from _debug_utils.im_viz import show_collection
import numpy as np
from pycv.colors import rgb2gray
from pycv.segmentation import im_threshold

# rng = np.random.default_rng()
# inputs = np.zeros((256, 256))
# inputs[64:-64, 64:-64] = 1
# inputs += 0.2 * rng.random(inputs.shape)

# inputs = load_image('lena.jpg')
#
# output_b = bilinear_resize(inputs, 300, 400, axis=(0, 1))
# output_nn = nearest_neighbour_resize(inputs, 300, 400, axis=(0, 1))
#
# show_collection([inputs, output_b, output_nn], 1, 3)

inputs = load_image('coins.png')
bin_image, th = im_threshold(
    inputs, 'otsu',
)

show_collection([inputs, bin_image], 1, 2)
