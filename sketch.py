from _debug_utils.im_load import load_image, load_defualt_binary_image
from _debug_utils.im_viz import show_collection
import numpy as np
from pycv._lib.core import ops
from pycv.draw import draw_line, draw_ellipse, draw_circle


inputs = np.zeros((25, 25), bool)
inputs = draw_circle((12, 12), 3, output=inputs)

show_collection([inputs], 1, 1)


# rng = np.random.default_rng()
# inputs = np.zeros((256, 256))
# inputs[64:-64, 64:-64] = 1
# inputs += 0.2 * rng.random(inputs.shape)

# inputs = load_image('lena.jpg')
# show_collection([inputs, output_b, output_nn], 1, 3)
