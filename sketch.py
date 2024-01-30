# from _debug_utils.im_load import load_image, load_defualt_binary_image
# from _debug_utils.im_viz import show_collection, show_struct
import numpy as np
from pycv._lib.core import ops
from pycv._lib.core_support import interpolation_py as intrp

inputs = np.zeros((15, 15), np.float64) + 1
# inputs = draw_line((1, 7), (13, 7), output=inputs)

# inputs[2:-2, 2:-2] = 1

# inputs = cube(15).astype(np.float64)

o = intrp.rotate(inputs, 45, order=0, reshape=True)


# rng = np.random.default_rng()
# inputs = np.zeros((256, 256))
# inputs[64:-64, 64:-64] = 1
# inputs += 0.2 * rng.random(inputs.shape)

# inputs = load_image('lena.jpg')
# show_collection([inputs, output_b, output_nn], 1, 3)
