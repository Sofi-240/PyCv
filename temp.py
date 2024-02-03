from _debug_utils.im_load import load_defualt_binary_image
from _debug_utils.im_viz import show_collection
import numpy as np
from pycv.draw import draw_line
from pycv.transform import resize
import functools


class Graham_Scan(object):
    def __init__(self):
        self.convex_hull = []
        self.arr_shape = None

    def __len__(self):
        return len(self.convex_hull)

    def cross_product(self, p1, p2, p3):
        p12 = p2 - p1
        p13 = p3 - p1
        return np.cross(p13, p12)

    def direction(self, p1, p2, p3):
        o = self.cross_product(p1, p2, p3)
        if o == 0:
            return 0
        if o < 0:
            return -1
        return 1

    def distance(self, p1, p2):
        return np.linalg.norm(p2 - p1)

    def cmp_points(self, p1, p2, P):
        ori = self.cross_product(P, p1, p2)
        if ori < 0:
            return -1
        if ori > 0:
            return 1
        if self.distance(P, p1) < self.distance(P, p2):
            return -1
        return 1

    def build(self, arr):
        self.arr_shape = arr.shape
        points = [p for p in np.stack(np.where(arr != 0), axis=1)]

        p0 = points[0]
        points = points[1:]
        sorted_polar = sorted(points, key=functools.cmp_to_key(lambda po1, po2: self.cmp_points(po1, po2, p0)))

        to_remove = []
        for i in range(len(sorted_polar) - 1):
            d = self.direction(p0, sorted_polar[i], sorted_polar[i + 1])
            if d == 0:
                to_remove.append(i)
        sorted_polar = [i for j, i in enumerate(sorted_polar) if j not in to_remove]

        stack = []
        stack.append(p0)
        stack.append(sorted_polar[0])
        stack.append(sorted_polar[1])
        stack_size = 3

        for i in range(2, len(sorted_polar)):
            while True:
                d = self.direction(stack[stack_size - 2], stack[stack_size - 1], sorted_polar[i])
                if d < 0:
                    break
                else:
                    stack.pop()
                    stack_size -= 1
            stack.append(sorted_polar[i])
            stack_size += 1

        self.convex_hull = [stack[0]] + sorted(stack[1:], key=functools.cmp_to_key(lambda po1, po2: self.cmp_points(po1, po2, p0)))

    def is_inside(self, point):
        if not len(self):
            return False
        for p1, p2 in zip(self.convex_hull, self.convex_hull[1:]):
            if self.cross_product(p1, p2, point) > 0:
                return False
        return True

    def mark_convex(self):
        if not len(self):
            return None
        mask = np.zeros(self.arr_shape, np.uint8)
        for p1, p2 in zip(self.convex_hull, self.convex_hull[1:]):
            mask[draw_line(tuple(p1), tuple(p2))] = 255

        mask[draw_line(tuple(self.convex_hull[-1]), tuple(self.convex_hull[0]))] = 255
        return mask

    def fill_convex(self):
        if not len(self):
            return None
        mask = np.zeros(self.arr_shape, np.uint8)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                point = np.array([i, j], np.int64)
                if self.is_inside(point):
                    mask[i, j] = 1
        return mask


gs = Graham_Scan()
image = load_defualt_binary_image()
image = resize(image, (114, 200), order=0)

gs.build(image)

show_collection([image, gs.mark_convex(), gs.fill_convex()], 1, 3)

# Jarvis's March
# points = np.stack(np.where(image == 1), axis=1)
#
# p0 = points[0]
#
# points = [p for p in points]
#
# index = 0
# l = index
# result = [p0]
#
# while True:
#     q = (l + 1) % len(points)
#     for i in range(len(points)):
#         if i == l:
#             continue
#         if cmp_points(points[i], points[q], points[l]) > 0:
#             q = i
#     l = q
#     if l == index:
#         break
#     result.append(points[q])
#
# mask = np.zeros(image.shape, np.uint8)
# for p1, p2 in zip(result, result[1:]):
#     mask[draw_line(tuple(p1), tuple(p2))] = 255
#
# mask[draw_line(tuple(result[-1]), tuple(result[0]))] = 255
#
# show_collection([image, mask], 1, 2)