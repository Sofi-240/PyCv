from _debug_utils.im_load import load_image, load_defualt_binary_image
from _debug_utils.im_viz import show_collection
import numpy as np
from pycv._lib.core import ops
from pycv._lib.core_support.convexhull_py import convex_hull_2d
from pycv.draw import draw_line, draw_circle
from pycv._lib.filters_support.windows import diamond, circle
from pycv.morphological import im_label

# image = diamond(7).astype(np.uint8)

image = np.zeros((30, 30), np.uint8)
image[1:16, 1:16] = circle(7).astype(np.uint8)
image[15:, 15:] = diamond(7).astype(np.uint8)
# image = load_defualt_binary_image()

