# from _debug_utils.im_load import load_image, load_defualt_binary_image
# from _debug_utils.im_viz import show_collection
import numpy as np
from pycv._lib.core import ops
from pycv.draw import draw_line


img = np.zeros((15, 15))
img[draw_line((1, 13), (13, 1))] = 255
img[draw_line((1, 1), (13, 13))] = 255


thetas = np.linspace(-90, 90, 360, endpoint=False) * np.pi / 180.0

offset = int(np.ceil(np.hypot(img.shape[0], img.shape[1])))
max_distance = 2 * offset + 1

h_space = np.zeros((max_distance, thetas.shape[0]), dtype=np.uint64)
rho = np.linspace(-offset, offset, max_distance)

ops.hough_transform(img, thetas, h_space)




# show_collection([img, np.log(1 + h_space)], 1, 2)


# rng = np.random.default_rng()
# inputs = np.zeros((256, 256))
# inputs[64:-64, 64:-64] = 1
# inputs += 0.2 * rng.random(inputs.shape)

# inputs = load_image('lena.jpg')
# show_collection([inputs, output_b, output_nn], 1, 3)
