from pycv.io import ImageLoader, show_collection, DEFAULT_DATA_PATH
from pycv.filters import canny

########################################################################################################################

loader = ImageLoader(DEFAULT_DATA_PATH)

lena = loader.load('lena', _color_fmt='RGB2GRAY')

edge = canny(lena, sigma=1.1)

show_collection([lena, edge])

########################################################################################################################
