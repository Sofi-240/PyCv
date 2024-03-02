import numpy as np
from _debug_utils.im_load import load_image
from _debug_utils.im_viz import show_collection
from pycv._lib._src import c_pycv
from pycv.filters import median_filter
from pycv.segmentation import im_threshold
from pycv.morphological import binary_edge, binary_dilation
from pycv.transform import hough_circle
from pycv.draw import draw_circle
from pycv._lib._src_py.pycv_measure import find_object_peaks
from pycv._lib._src_py.kdtree import KDtree
from scipy.spatial import cKDTree

coins = load_image('coins.png')[160:230, 70:250]
coins_med = median_filter(coins, (5, 5))
coins_bin, th = im_threshold(coins_med, 'otsu')
inputs = binary_edge(coins_bin, 'outer')
radius = np.arange(20, 35, 2)

h_space = hough_circle(inputs, radius)

peaks_mask = find_object_peaks(h_space, (1, 1))
peaks = np.where(peaks_mask)

sorted_ = np.argsort(h_space[peaks])[::-1]
peaks = tuple(p[sorted_] for p in peaks)

peaks_radius = radius[peaks[0]]
peaks_cc = np.stack(peaks[1:], axis=1)
peaks_h = h_space[peaks]

tree = KDtree(peaks_cc, 1)

t2 = cKDTree(peaks_cc, leafsize=1, balanced_tree=True)

query_nn = tree.query_ball_point(peaks_cc,  np.hypot(7, 7))

mask = np.ones_like(peaks_radius, bool)

for ii, nn in enumerate(query_nn):
    if mask[ii]:
        for jj in nn:
            if jj != ii:
                mask[jj] = 0

hh_bin = np.zeros_like(coins_bin)

for ii in mask.nonzero()[0]:
    hh_bin[draw_circle(tuple(int(i) for i in peaks_cc[ii]), int(peaks_radius[ii]))] = 1


hh_bin = binary_dilation(hh_bin)

marked = np.zeros(hh_bin.shape + (3, ), bool)
marked[..., 0] = coins_bin | hh_bin
marked[..., 1] = coins_bin & ~hh_bin
marked[..., 2] = marked[..., 1]

marked = marked.astype(np.uint8) * 255

show_collection([coins_bin, marked], 1, 2)