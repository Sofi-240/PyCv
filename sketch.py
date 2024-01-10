import numpy as np
from _debug_utils.im_load import load_image
from _debug_utils.im_viz import show_collection
from pycv.filters.generic import median_filter, mean_filter
from pycv._lib.array_api.dtypes import cast

# image = (255 - load_image('horse.png')[..., 0] > 1)

image = load_image('camera.png')

s = mean_filter(image, 7, preserve_dtype=True)

show_collection([image, s], 1, 2)



