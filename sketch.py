from _debug_utils.im_load import load_image
from _debug_utils.im_viz import show_collection
from pycv.colors.color import rgb2gray
import numpy as np
from pycv._lib.filters_support.canny_edge import canny_filter

rng = np.random.default_rng()
inputs = np.zeros((256, 256))
inputs[64:-64, 64:-64] = 1
inputs += 0.2 * rng.random(inputs.shape)

# inputs = rgb2gray(load_image('lena.jpg'))

strong_edge = canny_filter(inputs, sigma=1.4)


show_collection([inputs, strong_edge], 1, 2)

