import numpy as np
import os.path as osp
from pycv._lib._src import c_pycv
from pycv.io import ImageLoader, show_collection
from pycv.morphological import binary_edge, ConvexHull

########################################################################################################################

loader = ImageLoader(osp.join(osp.dirname(__file__), '_debug_utils', 'data'))

horse = (255 - loader.load('horse')[..., 0]) > 0
edge = binary_edge(horse)

chull = ConvexHull(image=edge)

im = chull.to_image(edge.shape)

show_collection([edge, im])