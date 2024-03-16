import numpy as np
import os.path as osp
from pycv._lib._src import c_pycv
from pycv.io import ImageLoader, show_collection

########################################################################################################################

loader = ImageLoader(osp.join(osp.dirname(__file__), '_debug_utils', 'data'))

horse = loader.load('horse')[..., 0] == 0

