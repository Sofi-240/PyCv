import numpy as np
from _debug_utils.im_load import load_image
from pycv.colors import rgb2gray
from pycv._indo._pyramids import GaussianPyramid, LaplacianPyramid, DOGPyramid, GaussianScaleSpace
from _debug_utils.im_viz import show_collection

image = rgb2gray(load_image('astronaut.png'))

pyramid = GaussianScaleSpace(image)

out = [p for p in pyramid]

show_collection(out[4], 2, 3)

# a = [aa for aa in out[0]]
