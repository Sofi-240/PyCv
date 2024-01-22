from _debug_utils.im_load import load_image, load_defualt_binary_image
from _debug_utils.im_viz import show_collection
import numpy as np
from pycv.segmentation import im_threshold
from pycv.morphological import im_label, remove_small_holes, remove_small_objects
from pycv._lib.core import ops


# rng = np.random.default_rng()
# inputs = np.zeros((256, 256))
# inputs[64:-64, 64:-64] = 1
# inputs += 0.2 * rng.random(inputs.shape)

# inputs = load_image('lena.jpg')
# show_collection([inputs, output_b, output_nn], 1, 3)

inputs = load_image('coins.png')[85:, :]

bin_image, th = im_threshold(inputs, 'kapur')

clean = remove_small_holes(bin_image, 500)
clean = remove_small_objects(clean, 100)

show_collection([inputs, bin_image, clean], 1, 3)
