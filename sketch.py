import numpy as np
from pycv._lib._src import c_pycv
from pycv.features import HaarFeatures, HaarType


feature = HaarFeatures(HaarType.HAAR_LINE, axis=0, feature_dims=(5, 5))
coord = feature.coordinates()
ff = feature.haar_like_feature(np.ones((11, 11), np.uint8), integrate=True)

