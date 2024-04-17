import numpy as np
import os.path as osp
from pycv.io import ImageLoader, show_collection, DEFAULT_DATA_PATH
from pycv.features import harris_corner, find_peaks
from pycv.draw import mark_points, draw_ellipse
from pycv.filters import adjust_gamma

########################################################################################################################

loader = ImageLoader(DEFAULT_DATA_PATH)

brick = loader.load('brick', _color_fmt='RGB2GRAY')

brick = adjust_gamma(brick, gamma=2)

corners = harris_corner(brick)
peaks = find_peaks(corners, min_distance=7, threshold=0.8)
marked = mark_points(brick, np.stack(np.where(peaks), axis=-1))
show_collection([brick, marked])