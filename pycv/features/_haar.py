import numpy as np
from .._lib.array_api.regulator import np_compliance
from pycv._lib._src.c_pycv import CHaarFeatures
from ._features import integral_image
from pycv._lib._members_struct import Members

__all__ = [
    "HaarType",
    "HaarFeatures",
]


########################################################################################################################


class HaarType(int, Members):
    HAAR_EDGE = 1
    HAAR_LINE = 2
    HAAR_DIAG = 3


def _valid_axis(htype: HaarType, axis: int) -> tuple:
    if htype == HaarType.HAAR_EDGE or htype == HaarType.HAAR_LINE:
        out = (axis,)
        return out
    return (0, 1)


class HaarFeatures(CHaarFeatures):
    def __init__(self, htype: HaarType, axis: int, feature_dims: tuple):
        axis = _valid_axis(htype, axis)
        if len(feature_dims) != 2:
            raise TypeError("currently haar features supported just for 2D arrays")
        super().__init__(int(htype), 2, feature_dims, axis)

    def coordinates(self, *, dims: tuple | None = None):
        if dims is not None:
            if len(dims) != self.ndim:
                raise TypeError("dims size need to be equal to ndim")
        else:
            dims = self.feature_dims
        return super().coordinates(dims=dims)

    def haar_like_feature(self, integral: np.ndarray, top_left: tuple | None = None, integrate: bool = False):
        kw = {}
        if top_left is not None:
            if len(top_left) != self.ndim:
                raise TypeError("top left size need to be equal to ndim")
            if any(tp >= s for tp, s in zip(top_left, integral.shape)):
                raise RuntimeError("top left point is out of range ")
        else:
            top_left = (0, ) * self.ndim
        integral = np_compliance(integral, arg_name='integral', _check_finite=True)
        if integral.ndim > self.ndim:
            raise ValueError('integral need to have same ndim as the feature')
        if not integrate:
            return super().haar_like_feature(integral, **kw)
        inputs = integral_image(integral)
        return super().haar_like_feature(inputs, top_left=top_left)

########################################################################################################################

