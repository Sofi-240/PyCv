from _debug_utils.im_load import load_image, load_defualt_binary_image
from _debug_utils.im_viz import show_collection
import numpy as np
from pycv._lib.core import ops
from pycv.draw import draw_line, draw_circle
from pycv._lib.filters_support.windows import diamond


image = diamond(7).astype(np.uint8)

# image = load_defualt_binary_image()

convex = ops.convex_hull(1, image, None)

mask = np.zeros(image.shape, np.uint8)
for p1, p2 in zip(convex, convex[1:]):
    mask[draw_line(tuple(p1), tuple(p2))] = 255

mask[draw_line(tuple(convex[-1]), tuple(convex[0]))] = 255

show_collection([image, mask], 1, 2)

# point = np.array([0, 0], np.int64)
# for p1, p2 in zip(convex, convex[1:]):
#     if np.cross(p2 - p1, point - p1) < 0:
#         print(False)
#         break

