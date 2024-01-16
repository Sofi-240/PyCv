import numpy as np
from pycv._lib.core import ops
from _debug_utils.im_load import load_defualt_binary_image
from _debug_utils.im_viz import show_collection
from pycv.morphological.binary import skeletonize

inputs = load_defualt_binary_image()

output = skeletonize(inputs)

show_collection([inputs, output], 1, 2)

