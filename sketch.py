import numpy as np
from _debug_utils.im_load import load_image
from _debug_utils.im_viz import show_collection
from pycv._lib.array_api.dtypes import cast
from pycv._lib.core import ops

image = (255 - load_image('horse.png')[..., 0] > 1)
kernel = np.ones((3, 3), bool)
origins = tuple(s // 2 for s in kernel.shape)
output = np.zeros_like(image)

ops.binary_erosion(image, kernel, output, origins, 1, None, 0)
edge = image ^ output

fill = edge.copy()

kernel[1, 1] = 0

ops.binary_region_fill(fill, kernel, (128, 119) , origins)

show_collection([fill, edge], 1, 2)


# show_collection([image, edge], 1, 2)
# image = load_image('camera.png')

# show_collection([image, s], 1, 2)
# dtype = np.uint8
# image = np.zeros((50, 50), dtype)
# image[2:4, 2:4] = 1
# image[5:10, 5:10] = 1

# kernel = np.ones((3, 3), bool)

# origins = tuple(s // 2 for s in kernel.shape)
# output = np.zeros(tuple(s - (2 * o) for s, o in zip(image.shape, origins)), dtype)

# output = np.zeros_like(image)
# mask = np.ones_like(image)
# mask[2:4, 2:4] = 0

# values = np.ones(kernel.shape, dtype) * 10

# ops.convolve(image, values, output, origins)
# ops.binary_erosion(image, kernel, output, origins, 1, mask, 0)
# ops.erosion(image, kernel, values, output, origins, None, 0)
# ops.dilation(image, kernel, None, output, origins, mask, 255)
