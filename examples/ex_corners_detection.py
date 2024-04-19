import numpy as np
import os.path as osp
from pycv.io import ImageLoader, show_collection, DEFAULT_DATA_PATH
from pycv.features import harris_corner, find_peaks, corner_fast
from pycv.draw import mark_points, draw_ellipse
from pycv.filters import adjust_gamma
from pycv.segmentation import Thresholds

########################################################################################################################

loader = ImageLoader(DEFAULT_DATA_PATH)

brick = loader.load('brick', _color_fmt='RGB2GRAY')

brick = adjust_gamma(brick, gamma=2)

harris = harris_corner(brick)
harris_peaks = find_peaks(harris, min_distance=7, threshold=0.8)
harris_marked = mark_points(brick, np.stack(np.where(harris_peaks), axis=-1))

fast = corner_fast(brick, n=12, threshold=0.15 * 255)
fast_peaks = find_peaks(fast, min_distance=11, threshold=Thresholds.MINIMUM(fast) * 2)
fast_marked = mark_points(brick, np.stack(np.where(fast_peaks), axis=-1))

show_collection([brick, harris_marked, fast_marked])

########################################################################################################################
