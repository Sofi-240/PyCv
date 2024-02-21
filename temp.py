import numpy as np
from _debug_utils.im_load import load_image
# from _debug_utils.im_viz import show_collection
from pycv._lib._src import c_pycv
from pycv.draw import draw_line, draw_circle, draw_ellipse
from pycv.filters import median_filter, rank_filter, local_max_filter
from pycv.segmentation import im_threshold
from pycv.morphological import im_label, find_object, binary_edge
from pycv.transform import hough_circle, hough_line
from pycv._lib._src_py.pycv_measure import find_object_peaks

coins = load_image('coins.png')[160:230, 70:250]
coins_med = median_filter(coins, (5, 5))
coins_bin, th = im_threshold(coins_med, 'otsu')
inputs = binary_edge(coins_bin, 'outer')
radius = np.arange(20, 35, 2)

h_space = hough_circle(inputs, radius)

peaks = find_object_peaks(h_space, (1, 1))
peaks_cc = np.stack(np.where(peaks), axis=1)


# inputs = np.zeros((15, 15), np.uint8)
# inputs[draw_line((1, 1), (13, 13))] = 1
# inputs[draw_line((1, 7), (13, 7))] = 1
# inputs[draw_line((1, 13), (13, 1))] = 1
#
# h_space, theta, dist = hough_line(inputs)
# h_space2, _, _ = hough_line(np.array([inputs, inputs]))
# h_space3, _, _ = hough_line(inputs)
#
# print(np.sum(h_space2[0] != h_space))
# print(np.sum(h_space3 != h_space))


# inputs = np.zeros((25, 25), bool)
# inputs[draw_circle((12, 12), 7)] = 1
# inputs[draw_circle((7, 7), 5)] = 1
# radius = np.arange(2, 10)