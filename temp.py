from _debug_utils.im_load import load_defualt_binary_image
import numpy as np
from numpy.testing import assert_array_almost_equal
from pycv._lib.array_api.array_pad import get_padding_width, pad
from pycv.colors import rgb2gray
from pycv.draw import draw_line, draw_circle
from pycv._lib._src import c_pycv


# inputs = np.zeros((11, 11), dtype=np.uint8)
# inputs[draw_line((1, 1), (9, 9))] = 1
# inputs[draw_circle((5, 5), 4)] = 1







