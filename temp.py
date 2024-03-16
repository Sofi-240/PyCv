import numpy as np
import os.path as osp
from pycv._lib._src import c_pycv
from pycv.io import ImageLoader, show_collection
from pycv.segmentation import Thresholds, im_threshold
from pycv.morphological import im_label, find_object, gray_closing, Strel

########################################################################################################################

loader = ImageLoader(osp.join(osp.dirname(__file__), '_debug_utils', 'data'))

coins = loader.load('coins')

coins_bin, th = im_threshold(gray_closing(coins, Strel.SQUARE(5)), Thresholds.OTSU)

n_labels, labels = im_label(coins_bin)

show_collection([coins, coins_bin, labels.astype(np.uint8)])

########################################################################################################################







