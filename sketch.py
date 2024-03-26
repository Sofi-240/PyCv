import numpy as np
import os.path as osp
from pycv._lib._src import c_pycv
from pycv._lib._src_py import pycv_morphology
from pycv.draw import draw_circle, draw_ellipse
from pycv.transform import rotate
from pycv.measurements._label_properties import Bbox, RegionProperties
from pycv.io import show_collection
from pycv.morphological import im_label, find_object, region_fill, binary_edge, ConvexHull, binary_fill_holes
from skimage.measure import regionprops, inertia_tensor
import itertools

########################################################################################################################

circles = np.zeros((50, 50), bool)

draw_ellipse((25, 25), 20, 10, circles)
region_fill(circles, (25, 25), inplace=True)

rot = rotate(circles, 45).astype(bool)

n_labels, labels = im_label(rot)

boxes = [Bbox(bbox) for bbox in find_object(labels, as_slice=True)]
box = boxes[0]

region_props = RegionProperties(box(labels == 1), offset=box, label=1)
f = region_props.filled_image

# region_props_sk = regionprops(labels, offset=box.top_left)[0]

########################################################################################################################

