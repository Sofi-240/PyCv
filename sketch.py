from _debug_utils.im_load import load_image, load_defualt_binary_image
from _debug_utils.im_viz import show_collection
import numpy as np
from pycv._lib.core import ops
from pycv.draw import draw_line, draw_circle
from pycv.transform import resize


def cross_product(_p0, _p1, _p2):
    return np.cross(_p2 - _p0, _p1 - _p0)


def distance(_p1, _p2):
    return np.linalg.norm(_p2 - _p1, ord=1)


def cmp_points(_p0, _p1, _p2):
    ori = cross_product(_p0, _p1, _p2)
    if ori < 0:
        return -1
    if ori > 0:
        return 1
    if distance(_p0, _p1) < distance(_p0, _p2):
        return -1
    return 1


def heapify(_points, n, i, _p0):
    # Find largest among root and children
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and cmp_points(_p0, _points[i], _points[l]) < 0:
        largest = l

    if r < n and cmp_points(_p0, _points[largest], _points[r]) < 0:
        largest = r

    # If root is not largest, swap with largest and continue heapifying
    if largest != i:
        _points[i], _points[largest] = _points[largest], _points[i]
        heapify(_points, n, largest, _p0)


def heapSort(_points, _p0):
    n = len(_points)

    # Build max heap
    for i in range(n // 2, -1, -1):
        heapify(_points, n, i, _p0)

    for i in range(n - 1, 0, -1):
        # Swap
        _points[i], _points[0] = _points[0], _points[i]

        # Heapify root element
        heapify(_points, i, 0, _p0)


# image = np.array(
#     [[0, 0, 1, 0, 0],
#      [0, 1, 0, 1, 0],
#      [1, 0, 0, 0, 1],
#      [0, 1, 0, 1, 0],
#      [0, 0, 1, 0, 0]], np.uint8
# )

image = load_defualt_binary_image()

convex = ops.convex_hull(image, None)

mask = np.zeros(image.shape, np.uint8)
for p1, p2 in zip(convex, convex[1:]):
    mask[draw_line(tuple(p1), tuple(p2))] = 255

mask[draw_line(tuple(convex[-1]), tuple(convex[0]))] = 255

show_collection([image, mask], 1, 2)

