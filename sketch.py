from _debug_utils.im_load import load_image
from _debug_utils.im_viz import show_collection
import numpy as np
from pycv.filters import canny
from pycv.colors import rgb2gray

# rng = np.random.default_rng()
# inputs = np.zeros((1000, 1000))
# inputs[64:-64, 64:-64] = 1
# inputs += 0.2 * rng.random(inputs.shape)

inputs = rgb2gray(load_image('lena.jpg'))

strong_edge = canny(inputs, sigma=1)

show_collection([inputs, strong_edge], 1, 2)

from pycv import filters





