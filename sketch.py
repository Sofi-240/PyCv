import numpy as np
from pycv._lib.core import ops
from pycv._lib.filters_support.windows import edge_kernel, SOBEL_EDGE, SOBEL_WEIGHTS

inputs = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0],
     [0, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0],
     [0, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0],
     [0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0],
     [0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
     [0, 3, 3, 3, 0, 3, 0, 0, 0, 0, 0],
     [0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.float64
)


# dy_kernel = np.flip(kernel, (0, 1))



