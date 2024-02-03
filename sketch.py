# from _debug_utils.im_load import load_image, load_defualt_binary_image
# from _debug_utils.im_viz import show_collection
import numpy as np
from pycv._lib.core import ops
from pycv.draw import draw_line, draw_circle
from pycv.transform import resize
import functools


def cmp_to_key(cmp_func):
    class K(object):
        def __init__(self, obj, *args):
            print('obj created with ', obj)
            self.obj = obj

        def __lt__(self, other):
            print('comparing less than ', self.obj)
            return cmp_func(self.obj, other.obj) < 0

        def __gt__(self, other):
            print('comparing greter than ', self.obj)
            return cmp_func(self.obj, other.obj) > 0

        def __eq__(self, other):
            print('comparing equal to ', self.obj)
            return cmp_func(self.obj, other.obj) == 0

        def __le__(self, other):
            print('comparing less than equal ', self.obj)
            return cmp_func(self.obj, other.obj) <= 0

        def __ge__(self, other):
            print('comparing greater than equal', self.obj)
            return cmp_func(self.obj, other.obj) >= 0

        def __ne__(self, other):
            print('comparing not equal ', self.obj)
            return cmp_func(self.obj, other.obj) != 0

    return K


# print(np.array2string(np.zeros((5, 5), np.uint8), separator=', '))
def my_cmp(x, y):
    print("compare ", x, " with ", y)
    if x > y:
        return 1
    elif x < y:
        return -1
    else:
        return 0


def mergeSort(array):
    if len(array) > 1:

        #  r is the point where the array is divided into two subarrays
        r = len(array) // 2
        L = array[:r]
        M = array[r:]

        # Sort the two halves
        mergeSort(L)
        mergeSort(M)

        i = j = k = 0

        # Until we reach either end of either L or M, pick larger among
        # elements L and M and place them in the correct position at A[p..r]
        while i < len(L) and j < len(M):
            if my_cmp(L[i], M[j]) < 0:
                array[k] = L[i]
                i += 1
            else:
                array[k] = M[j]
                j += 1
            k += 1

        # When we run out of elements in either L or M,
        # pick up the remaining elements and put in A[p..r]
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1

        while j < len(M):
            array[k] = M[j]
            j += 1
            k += 1


pp = [2, 3, 7, 1, 5, 0, 9]
a = sorted(pp, key=functools.cmp_to_key(my_cmp))

pp_s = pp[:]

mergeSort(pp_s)
