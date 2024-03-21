import numpy as np
from pycv._lib._src import c_pycv
from pycv.dsa import KMeans

########################################################################################################################


inputs = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
     [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    dtype=bool
)

points = np.stack(np.where(inputs), axis=-1).astype(np.float64)

clust = KMeans(points, k=4)
pred = clust.predict(points, pnorm=2)

pred_im = np.zeros_like(inputs, dtype=np.int64)
pred_im[inputs] = pred + 1
