from _debug_utils.im_load import load_image, load_defualt_binary_image
# from _debug_utils.im_viz import show_collection
import numpy as np
from pycv._lib.filters_support.windows import gaussian_kernel
from pycv._lib.core import ops
from pycv.colors import rgb2gray
from pycv._lib.core_support import interpolation_py as intrp
from pycv._lib.core_support.filters_py import convolve

inputs = rgb2gray(load_image('astronaut.png'))

# inputs_base = intrp.resize(inputs, tuple(s * 2 for s in inputs.shape), order=1, padding_mode='reflect')
#
# base_kernel = gaussian_kernel(1.24, ndim=2, truncate=3)
#
# tmp = convolve(inputs_base, base_kernel, padding_mode='reflect')

# tmp = convolve(inputs_base, base_kernel, axis=0, padding_mode='reflect')
# convolve(inputs_base, base_kernel, axis=1, output=inputs_base, padding_mode='reflect')


# rng = np.random.default_rng()
# inputs = np.zeros((256, 256))
# inputs[64:-64, 64:-64] = 1
# inputs += 0.2 * rng.random(inputs.shape)

# inputs = load_image('lena.jpg')
# show_collection([inputs, output_b, output_nn], 1, 3)
