import numpy as np
from _debug_utils.im_load import load_image
# from _debug_utils.im_viz import show_collection
from pycv._lib._src import c_pycv
from pycv.measurements import im_label, find_object
from pycv.segmentation import im_threshold
from pycv.filters import median_filter
from pycv._indo.regionprops import region_properties
from skimage.measure import regionprops

# coins = load_image('coins.png')[88:]
# coins_med = median_filter(coins, (5, 5))
# coins_bin, th = im_threshold(coins_med, 'otsu')
#
# # show_collection([coins_med, coins_bin], 1, 2)
#
# n_labels, labels = im_label(coins_bin, connectivity=2)
#
# objects = region_properties(labels, intensity_image=coins_med)
# obj = objects[0]
#
#
# a = regionprops(labels)


