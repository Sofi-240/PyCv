import numpy as np
from _debug_utils.im_load import load_image, load_defualt_binary_image
# from _debug_utils.im_viz import show_collection
from pycv._lib._src import c_pycv
from pycv.morphological import im_label, region_fill, convex_hull, binary_dilation, binary_fill_holes, gray_erosion
from pycv.filters import median_filter, local_min_filter
from pycv.segmentation import im_threshold
from pycv.draw import draw_circle
from skimage.transform import hough_line_peaks
from pycv.measurements._regionprops import region_properties, RegionProperties


inputs = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)

out = local_min_filter(inputs, footprint=footprint, padding_mode='reflect')

# inputs = np.zeros((11, 11), np.int64)
# inputs[draw_circle((5, 5), 4)] = 1
# inputs[draw_circle((5, 5), 3)] = 1
#
# obj = region_properties(inputs)[0]


# coins = load_image('coins.png')[88:]
# coins_med = median_filter(coins, (5, 5))
# coins_bin, th = im_threshold(coins_med, 'otsu')
#
# # show_collection([coins_med, coins_bin], 1, 2)
#
# n_labels, labels = im_label(coins_bin, connectivity=2)
#
# a = regionprops(labels, coins_med)
#
# objects = region_properties(labels, intensity_image=coins_med)
