import numpy as np
from pycv._lib._src import c_pycv
from pycv.features import glcm, glcm_props

########################################################################################################################

image = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [0, 2, 2, 2],
                  [2, 2, 3, 3]],
                 dtype=np.uint8)

distances = np.array([1, 2, 3], np.float64)
angle = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4], np.float64)
levels = 4

p = glcm(image, distances, angle, levels, normalize=True)

g_props = glcm_props(p)
