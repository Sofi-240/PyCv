import numpy as np
import os.path as osp
from types import DynamicClassAttribute
from pycv._lib._src import c_pycv
from pycv.draw import mark_points, Shapes, draw_circle
from pycv.measurements import Bbox, RegionProperties
from pycv.io import ImageLoader, show_collection
from pycv.segmentation import Thresholds, im_threshold
from pycv.morphological import im_label, find_object, gray_closing, Strel, remove_small_objects, region_fill, \
    convex_hull

########################################################################################################################

loader = ImageLoader(osp.join(osp.dirname(__file__), '_debug_utils', 'data'))

coins = loader.load('coins')

coins_bin, th = im_threshold(gray_closing(coins, Strel.SQUARE(5)), Thresholds.OTSU)

coins_bin = remove_small_objects(coins_bin)

n_labels, labels = im_label(coins_bin)

########################################################################################################################

boxes = [Bbox(bbox) for bbox in find_object(labels, as_slice=True)]
#
# props = [RegionProperties(bbox(labels), bbox(coins), offset=bbox, label=int(lbl)) for bbox, lbl in
#          zip(boxes, np.unique(labels[labels > 0]))]

# center = [(prop.centroid + 0.5).astype(np.int64) for prop in props]
#
# marked = mark_points(coins_bin, center, Shapes.CIRCLE, (255, 0, 0))
#
# show_collection([coins, coins_bin, marked])

########################################################################################################################
