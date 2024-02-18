import numpy as np
# from _debug_utils.im_viz import show_collection
from pycv._lib._src import c_pycv
from pycv.draw import draw_line
from pycv.filters import local_max_filter
from pycv.segmentation import im_binarize
from pycv.morphological import im_label, find_object
from pycv.measurements._peaks import find_object_peak

inputs = np.zeros((15, 15), np.uint8)
inputs[draw_line((1, 1), (13, 13))] = 1
inputs[draw_line((1, 7), (13, 7))] = 1
inputs[draw_line((1, 13), (13, 1))] = 1

theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
offset = int(np.ceil(np.hypot(inputs.shape[0], inputs.shape[1])))
dist = np.linspace(-offset, offset, 2 * offset + 1)

h_space = c_pycv.hough_transform(1, inputs, theta, offset=offset)
peaks = find_object_peak(h_space, min_distance=(9, 10), threshold=0.5 * np.max(h_space))

lines_start, lines_end = c_pycv.hough_transform(3, inputs, theta, offset=offset, threshold=10, line_length=8, line_gap=10)

