from _debug_utils.im_load import load_image
from pycv.colors import rgb2gray
# from _debug_utils.im_viz import show_collection

image = rgb2gray(load_image('astronaut.png'))


