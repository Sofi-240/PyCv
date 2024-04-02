import numpy as np
from pycv._lib._src import c_pycv
dtype = 2
dims = (5, 5)

feature = c_pycv.CHaarFeatures(dtype, len(dims), dims, [0])
coord = feature.coordinates()
integral = c_pycv.integral_image(np.ones((11, 11), np.uint8))
ff = feature.haar_like_feature(integral)


